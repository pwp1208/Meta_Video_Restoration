import argparse
import os
import time

class TestOptions :
    def __init__ (self) :
        self.parser = argparse.ArgumentParser ()
        self.initialized = False

    def initialize (self) :
        self.parser.add_argument ('--dataset', type=int, default= 0, help="Dataset name: 0: REVIDE, RainSynAll100, 1: haze, rain, rain_veil")
        self.parser.add_argument ('--test_dir', type=str, default="./video/REVIDE/", help="directory where testing videos are stored")
        self.parser.add_argument ('--pretrained_model_dir', type=str, default='pretrained_models', help='pretrained model are provided here')
        self.parser.add_argument ('--checkpoint_path', type=str, default='./ckpt/REVIDE/')
        self.parser.add_argument ('--output_path', type=str, default='./outputs/REVIDE/')
        self.parser.add_argument ('--frame_format', type=str, default='.jpg')
 
        # self.parser.add_argument ('--image_shape', type=str, default='256,256,3')
        # self.parser.add_argument('--test_num', type=int, default=-1)
        # self.parser.add_argument('--mode', type=str, default='test')
        self.initialized = True

    def parse (self) :
        if not self.initialized :
            self.initialize ()

        self.opt = self.parser.parse_args ()

        assert self.opt.dataset in [0, 1]
        
        if not os.path.isdir ('./outputs/') :
            os.mkdir ('./outputs/')

        # self.opt.testing_dir = os.path.join (self.opt.base_dir, self.opt.model_folder)

        # if not os.path.isdir (self.opt.testing_dir) :
        #     os.mkdir (self.opt.testing_dir)

        args = vars (self.opt)

        # print ("-"*20 + " Options " + "-"*20)
        # for k, v in sorted (args.items()) :
        #     print (str (k), ":", str (v))
        # print ("-"*20 + " End " + "-"*20)

        return self.opt