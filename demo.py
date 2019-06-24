import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable

#from FlowNet2 import FlowNet2
from models import FlowNet2
from utils.flowlib import flow_to_image
import matplotlib.pyplot as plt

import argparse

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
  parser.add_argument("--rgb_max", type=float, default=255.)
  args = parser.parse_args()

  # Prepare img pair
  im1 = imread('/mnt/data/FlyingChairs_examples/0000000-img0.ppm')
  im2 = imread('/mnt/data/FlyingChairs_examples/0000000-img1.ppm')
  # B x 3(RGB) x 2(pair) x H x W
  ims = np.array([[im1, im2]]).transpose((0, 4, 1, 2, 3)).astype(np.float32)
  ims = torch.from_numpy(ims)
  print(ims.size())
  ims_v = Variable(ims.cuda(), requires_grad=False)

  # Build model
  flownet2 = FlowNet2(args).cuda()
  #path = '/mnt/data/flownet2-pytorch/FlowNet2_checkpoint.pth.tar'
  path = '/home/tung/flownet2-pytorch/work/FlowNet2_model_best.pth.tar'
  pretrained_dict = torch.load(path)['state_dict']
  model_dict = flownet2.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  flownet2.load_state_dict(model_dict)
  flownet2.cuda()

  pred_flow = flownet2(ims_v).cpu().data
  pred_flow = pred_flow[0].numpy().transpose((1,2,0))
  flow_im = flow_to_image(pred_flow)

  # Visualization
  plt.imshow(flow_im)
  #plt.savefig('flow_trained_MPISintel.png', bbox_inches='tight')
  plt.savefig('flow_selftrained.png', bbox_inches='tight')
