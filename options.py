import argparse
import os
import time

class TestOptions :
    def __init__ (self) :
        self.parser = argparse.ArgumentParser ()
        self.initialized = False

    def initialize (self) :
        self.parser.add_argument ('--dataset', type=int, default= 0, help="Dataset name: 0: REVIDE, RainSynAll100, 1: Night-haze, Night-rain, Night-rain_veil")
        self.parser.add_argument ('--test_dir', type=str, default="./video/REVIDE/", help="directory where testing videos are stored")
        self.parser.add_argument ('--pretrained_model_dir', type=str, default='pretrained_models', help='pretrained model are provided here')
        self.parser.add_argument ('--checkpoint_path', type=str, default='./ckpt/REVIDE/')
        self.parser.add_argument ('--output_path', type=str, default='./outputs/REVIDE/')
        self.parser.add_argument ('--frame_format', type=str, default='.jpg')

        self.initialized = True

    def parse (self) :
        if not self.initialized :
            self.initialize ()

        self.opt = self.parser.parse_args ()

        assert self.opt.dataset in [0, 1]
        
        if not os.path.isdir ('./outputs/') :
            os.mkdir ('./outputs/')

        args = vars (self.opt)

        return self.opt