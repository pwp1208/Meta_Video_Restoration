
import os
import time
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display

from options import TestOptions
import datetime
import numpy as np
from termcolor import colored, cprint
import sys
import random
import glob
from PIL import Image

config = TestOptions().parse ()

def load(image_file):
    inputs_batch= []
    gt_batch = []
    
    for i in range(len(image_file)):
        image = tf.io.read_file(image_file[i])
        image = tf.image.decode_png(image)
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.BILINEAR) 
        GT_Enc = image[:, :w, :]
        GT_Enc = tf.image.resize(GT_Enc, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.BILINEAR)
        input_image = tf.expand_dims(input_image, axis=0)

        GT_Enc = tf.expand_dims(GT_Enc, axis=0)
        inputs_batch.append(input_image)
        gt_batch.append(GT_Enc)
  
    inputs_batch1 = tf.concat([inputs_batch[k] for k in range(len(inputs_batch))],axis=0)
    GT_Enc_batch1 = tf.concat([gt_batch[k] for k in range(len(gt_batch))],axis=0)

    inputs_batch1 = tf.cast(inputs_batch1, tf.float32)
    GT_Enc_batch1 = tf.cast(GT_Enc_batch1, tf.float32)

    return inputs_batch1, GT_Enc_batch1


BUFFER_SIZE = 142
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image



def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
    stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]



# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = ( input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_WIDTH, IMG_HEIGHT) ##### newly added
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image



##################### test video path ######################################################
test_task =sorted(glob.glob(config.test_dir),key = os.path.getmtime)

OUTPUT_CHANNELS = 3

##################### defined subblocks of proposed network ######################################################

def Conv_Block(filters, size, stride=1, dilation_rate=1,activation='',name=''):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',dilation_rate=dilation_rate,
                             kernel_initializer=initializer, name='_conv', use_bias=False))    
    result.add(tf.keras.layers.LeakyReLU())

    return result 

def DeConv_Block(filters, size, stride=2,name=''):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                    padding='same',
                                    kernel_initializer=initializer, name='_deconv',
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())
    return result  


def Conv_Activation(filters, size, stride=1,name=''):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                             kernel_initializer=initializer,  name=name+'_conv_th',activation='tanh'))
    return result  

filters = 32



def time_modulation( inp1, inp2, filt, name):
    avg_features = tf.keras.layers.Average()([inp1, inp2])
    conv_avg = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_avg')(avg_features)

    alpha = tf.sigmoid(tf.keras.layers.GlobalAveragePooling2D()(conv_avg))

    concat_feat = tf.concat([inp1, inp2], 3)
    conv_concat = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_concat')(concat_feat)

    beta = tf.keras.layers.GlobalAveragePooling2D()(conv_concat)

    multiplied_avg = tf.keras.layers.Multiply()([conv_concat, alpha])

    output = multiplied_avg + beta
    return output


def scale_modulation( inp1, inp2, filt, name):
    avg_features = tf.keras.layers.Average()([inp1, inp2])
    conv_avg = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_avg')(avg_features)

    alpha = tf.sigmoid(tf.keras.layers.GlobalAveragePooling2D()(conv_avg))

    concat_feat = tf.concat([inp1, inp2], 3)
    conv_concat = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + '_c_concat')(concat_feat)

    beta = tf.keras.layers.GlobalAveragePooling2D()(conv_concat)

    multiplied_avg = tf.keras.layers.Multiply()([conv_concat, alpha])

    output = multiplied_avg + beta
    return output


def multi_scale(inp1, filt, name=''):

    conv_1 = Conv_Block(filt, 1, stride=1, dilation_rate=1, name=name + 'c1')(inp1)
    conv_3 = Conv_Block(filt, 3, stride=1, dilation_rate=1, name=name + 'c3')(inp1)
    conv_5 = Conv_Block(filt, 5, stride=1, dilation_rate=1, name=name + 'c5')(inp1)

    return conv_1, conv_3, conv_5



def spatio_temporal_layer(inp1, inp2, filt, name = ''):

    c11, c13, c15 = multi_scale(inp1, filt, name = name + 'ms_inp1')
    c21, c23, c25 = multi_scale(inp2, filt, name = name + 'ms_inp2')

    sm_filt13_inp1 = scale_modulation( c11, c13, filt, name = name + 'sm_filt13_inp1')
    sm_filt15_inp1 = scale_modulation( c11, c15, filt, name = name + 'sm_filt15_inp1')
    sm_filt35_inp1 = scale_modulation( c13, c15, filt, name = name + 'sm_filt35_inp1')

    sm_filt13_inp2 = scale_modulation( c21, c23, filt, name = name + 'sm_filt13_inp2')
    sm_filt15_inp2 = scale_modulation( c21, c25, filt, name = name + 'sm_filt15_inp2')
    sm_filt35_inp2 = scale_modulation( c23, c25, filt, name = name + 'sm_filt35_inp2')

    tm1 = time_modulation( sm_filt13_inp1, sm_filt13_inp2, filt, name = name + 'tm1')
    tm2 = time_modulation( sm_filt15_inp1, sm_filt15_inp2, filt, name = name + 'tm2')
    tm3 = time_modulation( sm_filt35_inp1, sm_filt35_inp2, filt, name = name + 'tm3')

    concat_time = tf.concat([tm1,tm2,tm3],3)
    conv_concat_time = Conv_Block(filt, 1, stride=1, dilation_rate=1, name=name + 'concat_time')(concat_time)

    return conv_concat_time, c13, c23

def skip_prog_attention(in_prev, inp_current, in_up, filt, name = ''):
    up_in_prev = tf.keras.layers.UpSampling2D(size=(2, 2))(in_prev)
    up_in_up= tf.keras.layers.UpSampling2D(size=(2, 2))(in_up)

    concat_prev_current = tf.concat([up_in_prev, inp_current],3)
    conv_prev_current = Conv_Block(filt, 1, stride=1, dilation_rate=1, name=name + 'cat_prev_ct')(concat_prev_current)

    concat_in_up = tf.concat([up_in_up, conv_prev_current],3)
    conv_in_up = Conv_Block(filt, 1, stride=1, dilation_rate=1, name=name + 'cat_in_up')(concat_in_up)
    return conv_in_up


def Generator(name1 = ''):

        print(colored('################## Build generator 1 ##################','green'))    
        name1 = 'generator1'
        inputs1 = tf.keras.layers.Input(shape=[256, 256, 3])#self.inputs*1.0
        inputs2 = tf.keras.layers.Input(shape=[256, 256, 3])#self.inputs_t2*1.0
        in_32 = tf.keras.layers.Input(shape=[32, 32, filters*4])#self.inputs_t2*1.0
        in_64 = tf.keras.layers.Input(shape=[64, 64, filters*3])#self.inputs_t2*1.0
        in_128 = tf.keras.layers.Input(shape=[128, 128, filters*2])#self.inputs_t2*1.0
        in_256 = tf.keras.layers.Input(shape=[256, 256, filters])#self.inputs_t2*1.0
       
        # print(inputs.shape)
        conv_inp1 = Conv_Block(filters, 7, stride=1, dilation_rate=1, name=name1 + 'conv_inp1')(inputs1)
        conv_inp2 = Conv_Block(filters, 7, stride=1, dilation_rate=1, name=name1 + 'conv_inp2')(inputs2)
        en1_time_mod, en1_t1_conv3, en1_t2_conv3 = spatio_temporal_layer(conv_inp1, conv_inp2, filters, name = 'en1_layer')


        st_en1_t1_conv3 = Conv_Block(filters*2, 3, stride=2, dilation_rate=1, name=name1 + 'st_en1_t1_conv3')(en1_t1_conv3)
        st_en1_t2_conv3 = Conv_Block(filters*2, 3, stride=2, dilation_rate=1, name=name1 + 'st_en1_t2_conv3')(en1_t2_conv3)
        en2_time_mod, en2_t1_conv3, en2_t2_conv3 = spatio_temporal_layer(st_en1_t1_conv3, st_en1_t2_conv3, filters*2, name = 'en2_layer')


        st_en2_t1_conv3 = Conv_Block(filters*3, 3, stride=2, dilation_rate=1, name=name1 + 'st_en2_t1_conv3')(en2_t1_conv3)
        st_en2_t2_conv3 = Conv_Block(filters*3, 3, stride=2, dilation_rate=1, name=name1 + 'st_en2_t2_conv3')(en2_t2_conv3)
        en3_time_mod, en3_t1_conv3, en3_t2_conv3 = spatio_temporal_layer(st_en2_t1_conv3, st_en2_t2_conv3, filters*3, name = 'en3_layer')
        

        st_en3_t1_conv3 = Conv_Block(filters*4, 3, stride=2, dilation_rate=1, name=name1 + 'st_en3_t1_conv3')(en3_t1_conv3)
        st_en3_t2_conv3 = Conv_Block(filters*4, 3, stride=2, dilation_rate=1, name=name1 + 'st_en3_t2_conv3')(en3_t2_conv3)
        en4_time_mod, en4_t1_conv3, en4_t2_conv3 = spatio_temporal_layer(st_en3_t1_conv3, st_en3_t2_conv3, filters*4, name = 'en4_layer')
       
#########################################################decoder###############################################
        concat_t1_t2 = tf.concat([en4_t1_conv3, en4_t2_conv3],3)
        concat_t1_t2_conv = Conv_Block(filters*4, 3, stride=1, dilation_rate=1, name=name1 + 'concat_t1_t2_conv')(concat_t1_t2)
        

        skip1 = tf.concat([concat_t1_t2_conv, en4_time_mod], 3)
        conv_skip1 = Conv_Block(filters*4, 1, stride=1, dilation_rate=1, name=name1 + 'conv_skip1')(skip1)
        rec_cat1 = tf.concat([conv_skip1, in_32], 3)
        conv_rec_cat1 = Conv_Block(filters*4, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat1')(rec_cat1)
        dec_out1 = DeConv_Block(filters*3, 3, stride=2, name=name1+'deconv1')(conv_rec_cat1)#64*64


        attn_skip1 = skip_prog_attention(en4_time_mod, en3_time_mod, en4_time_mod, filters*3, name = 'attn1')
        skip2 = tf.concat([dec_out1, attn_skip1], 3)
        conv_skip2 = Conv_Block(filters*3, 1, stride=1, dilation_rate=1, name=name1 + 'conv_skip2')(skip2)
        rec_cat2 = tf.concat([conv_skip2, in_64], 3)
        conv_rec_cat2 = Conv_Block(filters*3, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat2')(rec_cat2)
        dec_out2 = DeConv_Block(filters*2, 3, stride=2, name=name1+'deconv2')(conv_rec_cat2)#128*128


        attn_skip2 = skip_prog_attention(en3_time_mod, en2_time_mod, attn_skip1, filters*2, name = 'attn2')
        skip3 = tf.concat([dec_out2, attn_skip2], 3)
        conv_skip3 = Conv_Block(filters*2, 1, stride=1, dilation_rate=1, name=name1 + 'conv_skip3')(skip3)
        rec_cat3 = tf.concat([conv_skip3, in_128], 3)
        conv_rec_cat3 = Conv_Block(filters*2, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat3')(rec_cat3)
        dec_out3 = DeConv_Block(filters, 3, stride=2, name=name1+'deconv3')(conv_rec_cat3)#256*256


        attn_skip3 = skip_prog_attention(en2_time_mod, en1_time_mod, attn_skip2, filters, name = 'attn3')
        skip4 = tf.concat([dec_out3, attn_skip3], 3)
        conv_skip4 = Conv_Block(filters, 1, stride=1, dilation_rate=1, name=name1 + 'conv_skip4')(skip4)
        rec_cat4 = tf.concat([conv_skip4, in_256], 3)
        conv_rec_cat4 = Conv_Block(filters, 1, stride=1, dilation_rate=1, name=name1 + 'rec_cat4')(rec_cat4)
        dec_out4 = DeConv_Block(3, 3, stride=1, name=name1+'deconv4')(conv_rec_cat4)#256*256
        
        Output = Conv_Activation(3, 1, stride=1, name=name1+ '_ConvAct_last')(dec_out4)   

         
        return tf.keras.Model(inputs=[inputs1,inputs2, in_32,in_64,in_128,in_256] , outputs=Output)


generator = Generator(name1 = 'meta')

print('Total Trainable parameters of meta_generator1 1 are :: {}'.format(generator.count_params()))

##################### Meta model ######################################################
generator = generator

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = config.checkpoint_path
if config.dataset == 0:
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
else:
    checkpoint_prefix = os.path.join(checkpoint_dir, "meta")

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)

if os.path.isdir (checkpoint_dir) :
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    print('Checkpoint Restored !!!!!')


##################### to save the training and validation results  ######################################################

try:
    os.mkdir(config.output_path)
except:
    pass

def generate_images(model, test_input, tar,mode='test'):
    in1 = tf.expand_dims(test_input,axis=0)
    results1 = []
    for i in range(test_input.shape[0]):
        if i ==0:
            # print(i)
            results_new_128 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(in1[:,i,:,:,:])
            results_new_64 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_128)
            results_new_32 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_64)
            result_new_256 = in1[:,i,:,:,:]

            result_256_1 = tf.concat([result_new_256,result_new_256,result_new_256,result_new_256,result_new_256,tf.expand_dims(result_new_256[:,:,:,0],axis = -1)],axis = 3)
            result_256 = tf.concat([result_256_1,result_256_1], axis=3)
            
            result_128_1 = tf.concat([results_new_128,results_new_128,results_new_128,results_new_128,results_new_128,tf.expand_dims(results_new_128[:,:,:,0],axis = -1)],axis=3)
            result_128 = tf.concat([result_128_1,result_128_1,result_128_1,result_128_1], axis=3)

            result_64_1 = tf.concat([results_new_64,results_new_64,results_new_64,results_new_64,results_new_64,tf.expand_dims(results_new_64[:,:,:,0],axis = -1)],axis=3)
            result_64 = tf.concat([result_64_1,result_64_1,result_64_1,result_64_1,result_64_1,result_64_1],axis =3)

            result_32_1  = tf.concat([results_new_32,results_new_32,results_new_32,results_new_32,results_new_32,tf.expand_dims(results_new_32[:,:,:,0],axis = -1)],axis=3)
            result_32 = tf.concat([result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1],axis = 3)

            output1 = model([in1[:,i,:,:,:],in1[:,i,:,:,:],result_32,result_64,result_128,result_256], training=True)

        elif i == 1:
            results_new_128 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results1[-1])
            results_new_64 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_128)
            results_new_32 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_64)
            result_new_256 = results1[-1]

            result_256_1 = tf.concat([result_new_256,result_new_256,result_new_256,result_new_256,result_new_256,tf.expand_dims(result_new_256[:,:,:,0],axis = -1)],axis = 3)
            result_256 = tf.concat([result_256_1,result_256_1], axis=3)
            
            result_128_1 = tf.concat([results_new_128,results_new_128,results_new_128,results_new_128,results_new_128,tf.expand_dims(results_new_128[:,:,:,0],axis = -1)],axis=3)
            result_128 = tf.concat([result_128_1,result_128_1,result_128_1,result_128_1], axis=3)

            result_64_1 = tf.concat([results_new_64,results_new_64,results_new_64,results_new_64,results_new_64,tf.expand_dims(results_new_64[:,:,:,0],axis = -1)],axis=3)
            result_64 = tf.concat([result_64_1,result_64_1,result_64_1,result_64_1,result_64_1,result_64_1],axis =3)

            result_32_1  = tf.concat([results_new_32,results_new_32,results_new_32,results_new_32,results_new_32,tf.expand_dims(results_new_32[:,:,:,0],axis = -1)],axis=3)
            result_32 = tf.concat([result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1],axis = 3)

            output1 = model([in1[:,i,:,:,:],in1[:,i-1,:,:,:],result_32,result_64,result_128,result_256], training=True)
     
        else:
            results_new_128 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results1[-1])
            results_new_64 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_128)
            results_new_32 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(results_new_64)
            result_new_256 = results1[-1]

            result_256_1 = tf.concat([result_new_256,result_new_256,result_new_256,result_new_256,result_new_256,tf.expand_dims(result_new_256[:,:,:,0],axis = -1)],axis = 3)
            result_256 = tf.concat([result_256_1,result_256_1], axis=3)
            
            result_128_1 = tf.concat([results_new_128,results_new_128,results_new_128,results_new_128,results_new_128,tf.expand_dims(results_new_128[:,:,:,0],axis = -1)],axis=3)
            result_128 = tf.concat([result_128_1,result_128_1,result_128_1,result_128_1], axis=3)

            result_64_1 = tf.concat([results_new_64,results_new_64,results_new_64,results_new_64,results_new_64,tf.expand_dims(results_new_64[:,:,:,0],axis = -1)],axis=3)
            result_64 = tf.concat([result_64_1,result_64_1,result_64_1,result_64_1,result_64_1,result_64_1],axis =3)

            result_32_1  = tf.concat([results_new_32,results_new_32,results_new_32,results_new_32,results_new_32,tf.expand_dims(results_new_32[:,:,:,0],axis = -1)],axis=3)
            result_32 = tf.concat([result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1,result_32_1],axis = 3)    

            output1 = model([in1[:,i,:,:,:],in1[:,i-2,:,:,:],result_32,result_64,result_128,result_256], training=True)

        results1.append(output1)

    outputs = results1[0]        

    for i in range(1,len(results1)):
        outputs = tf.concat([outputs , results1[i]], axis=0)
  
    for j in range(1,test_input.shape[0]):

    
        inp = Image.fromarray(np.array((test_input[j]*0.5 + 0.5)*255, dtype='uint8'))  
        out1 = Image.fromarray(np.array((outputs[j]*0.5 + 0.5)*255, dtype='uint8'))
        tar_1 = Image.fromarray(np.array((tar[j]*0.5 + 0.5)*255, dtype='uint8'))

        new_im = Image.new('RGB', (IMG_HEIGHT*1, IMG_WIDTH))
        try:
            os.mkdir(config.output_path + '/')
        except:
            pass
           
        x_offset = 0
        for im in [out1]:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

   
        new_im.save(config.output_path + '/'+str(j)+'.jpg')


n = 0
print("TESTING IMAGES ON SIZE {0}X{1}".format(IMG_HEIGHT,IMG_WIDTH))
for i,filename in enumerate(test_task):
    filenames1 = sorted(glob.glob(test_task[i] +'*'+config.frame_format),key =  os.path.getmtime)

    
    gen_input, target =load_image_train(filenames1)
    generate_images(generator, gen_input, target,  mode='test')
    print('n=',n,end='\r')
    n+=1 
print("TESTED SUCCESSFULLY")
