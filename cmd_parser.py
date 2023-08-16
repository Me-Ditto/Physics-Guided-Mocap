# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import configargparse

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'SEU-VCL diffusion project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='diffusion')

    parser.add_argument('--trainset',
                        default='',
                        type=str,
                        help='trainset.')
    parser.add_argument('--testset',
                        default='',
                        type=str,
                        help='testset.')
    parser.add_argument('--data_folder',
                        default='',
                        help='The directory that contains the data.')
    parser.add_argument('--keyp_folder',
                        default='',
                        help='The directory that contains the keypoints.')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--note',
                        default='test',
                        type=str,
                        help='code note')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate.')
    parser.add_argument('--batchsize',
                        default=10,
                        type=int,
                        help='batch size.')
    parser.add_argument('--frame_length',
                        default=16,
                        type=int,
                        help='frame length.')
    parser.add_argument('--num_joint',
                        default=24,
                        type=int,
                        help='num_joint.')
    parser.add_argument('--epoch',
                        default=500,
                        type=int,
                        help='num epoch.')
    parser.add_argument('--worker',
                        default=0,
                        type=int,
                        help='workers for dataloader.')
    parser.add_argument('--mode',
                        default='',
                        type=str,
                        help='running mode.')        
    parser.add_argument('--pretrain',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use pretrain parameters.')
    parser.add_argument('--use_vae_prior',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use pretrain parameters.')
    parser.add_argument('--vae_dir',
                        default='',
                        type=str,
                        help='vae_dir.')
    parser.add_argument('--pretrain_dir',
                        default='',
                        type=str,
                        help='The directory that contains the pretrain model.')
    parser.add_argument('--model_dir',
                        default='',
                        type=str,
                        help='(if test only) The directory that contains the model.')
    parser.add_argument('--offline_phys',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use offline phys results.')
    parser.add_argument('--phys_dir',
                        default='',
                        type=str,
                        help='The directory that contains the phys results.')
    parser.add_argument('--model',
                        default='',
                        type=str,
                        help='the model used for this project.')
    parser.add_argument('--train_loss',
                        default='L1 partloss',
                        type=str,
                        help='training loss type.')
    parser.add_argument('--test_loss',
                        default='L1',
                        type=str,
                        help='testing loss type.')
    parser.add_argument('--viz',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for visualize input.')
    parser.add_argument('--task',
                        default='ed_train',
                        type=str,
                        help='ee_train: encoder-encoder only, else ed_train.')
    parser.add_argument('--gpu_index',
                        default=0,
                        type=int,
                        help='gpu index.')
    parser.add_argument('--num_neurons',
                        default=512,
                        type=int,
                        help='num_neurons.')
    parser.add_argument('--latentD',
                        default=32,
                        type=int,
                        help='latentD.')
    parser.add_argument('--data_shape',
                        default=21,
                        type=int,
                        help='data_shape.')
    parser.add_argument('--kl_coef',
                        default=2e-4,
                        type=float,
                        help='kl_coef.')

    parser.add_argument('--timestep',
                        default=10,
                        type=int,
                        help='timestep.')
    
    # parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=1)
    # parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--checkpoint_epoch", type=int, default=0)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="data/sample_data/amass_copycat_take5_test_small.pkl")
    parser.add_argument("--vis_mode", type=str, default="all") # vis  ### mode
    parser.add_argument("--render_video", action="store_true", default=False)
    parser.add_argument("--render_rfc", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--hide_expert", action="store_true", default=False)
    parser.add_argument("--no_fail_safe", action="store_true", default=False)
    parser.add_argument("--focus", action="store_true", default=False)
    parser.add_argument("--phys_output", type=str, default="test")
    parser.add_argument("--shift_expert", action="store_true", default=False)
    parser.add_argument("--smplx", action="store_true", default=False)
    parser.add_argument("--hide_im", action="store_true", default=False)
    parser.add_argument("--adjust", action="store_true", default=False)


    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
