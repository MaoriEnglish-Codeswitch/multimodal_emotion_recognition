import torch
import numpy as np
import argparse
from data_utils import *
from torch.utils.data import DataLoader
from torchnet.dataset import TensorDataset
import train_tva_1
import random

if __name__ == '__main__':
    # get arguments
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)#1e-3
    p.add_argument('--rnndir', type=str, default=True,
                   help='Uni (False) or Bi (True) directional')
    p.add_argument('--rnnsize', type=int, default=60)#30)#200
    # video params
    p.add_argument('--vid_rnnnum', type=int, default=1)#1)#3
    p.add_argument('--vid_rnndp', type=int, default=0.5)#0.3
    p.add_argument('--vid_rnnsize', type=int, default=60)
    p.add_argument('--vid_nh', type=int, default=10,
                        help='number of attention heads for mha')#4
    p.add_argument('--vid_dp', type=int, default=0.1,
                        help='dropout rate for mha')#0.1
    # text params
    p.add_argument('--txt_rnnnum', type=int, default=1)
    p.add_argument('--txt_rnndp', type=int, default=0.5)#0.3
    p.add_argument('--txt_rnnsize', type=int, default=60)
    p.add_argument('--txt_nh', type=int, default=10,
                   help='number of attention heads for mha')#4
    p.add_argument('--txt_dp', type=int, default=0.1,
                   help='dropout rate for mha')#0.1
    # audio params
    p.add_argument('--aud_rnnnum', type=int, default=1)
    p.add_argument('--aud_rnndp', type=int, default=0.5)  # 0.3
    p.add_argument('--aud_rnnsize', type=int, default=60)
    p.add_argument('--aud_nh', type=int, default=10,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--aud_dp', type=int, default=0.1,
                   help='dropout rate for mha')  # 0.1
    # tv params
    p.add_argument('--tv_nh', type=int, default=10,
                   help='number of attention heads for mha')#4
    p.add_argument('--tv_dp', type=int, default=0.1,
                   help='dropout rate for mha')#0.1
    # ta params
    p.add_argument('--ta_nh', type=int, default=10,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--ta_dp', type=int, default=0.1,
                   help='dropout rate for mha')  # 0.1
    # vt params
    p.add_argument('--vt_nh', type=int, default=10,
                   help='number of attention heads for mha')#4
    p.add_argument('--vt_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # va params
    p.add_argument('--va_nh', type=int, default=10,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--va_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # at params
    p.add_argument('--at_nh', type=int, default=10,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--at_dp', type=int, default=0.1,
                   help='dropout rate for mha')
    # av params
    p.add_argument('--av_nh', type=int, default=10,
                   help='number of attention heads for mha')  # 4
    p.add_argument('--av_dp', type=int, default=0.1,
                   help='dropout rate for mha')

    p.add_argument('--output_dim', type=int, default=7,
                    help='number of classes')
    p.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')

    p.add_argument('--se_block_channels', type=int, default=6,
                    help='SE Block input channels')

    params = p.parse_args()
    #seed = 123
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(params.seed)

    # get train data

    x_text, x_vid, x_aud,labels = get_text_video_audio_data(params.data_path, 'train')
    train_dataset = TensorDataset([torch.Tensor(x_text).float().to('cuda'), torch.Tensor(x_vid).float().to('cuda'),
                                    torch.Tensor(x_aud).float().to('cuda'),torch.Tensor(labels).long().to('cuda')])
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    params.n_train = len(x_text)

    x_text, x_vid, x_aud, labels = get_text_video_audio_data(params.data_path, 'dev')

    dev_dataset = TensorDataset([torch.Tensor(x_text).float().to('cuda'), torch.Tensor(x_vid).float().to('cuda'),
                                  torch.Tensor(x_aud).float().to('cuda'),torch.Tensor(labels).long().to('cuda')])
    dev_loader = DataLoader(dev_dataset, batch_size=params.batch_size, shuffle=False)
    params.n_dev = len(x_text)

    x_text, x_vid, x_aud, labels = get_text_video_audio_data(params.data_path, 'test')

    test_dataset = TensorDataset([torch.Tensor(x_text).float().to('cuda'), torch.Tensor(x_vid).float().to('cuda'),
                                 torch.Tensor(x_aud).float().to('cuda'),torch.Tensor(labels).long().to('cuda')])
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    params.n_test = len(x_text)
    # train
    params.num_epochs = 30# give a random big number
    params.when = 10 # reduce LR patience

    test_loss = train_tva_1.initiate(params, train_loader, dev_loader, test_loader)
