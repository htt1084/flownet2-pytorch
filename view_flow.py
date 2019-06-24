import torch
import numpy as np 
import argparse

#from models import FlowNet2



import cvbase as cvb 


#flow_file = "/mnt/data/FlyingChairs_examples/0000000-gt.flo"
file_name = "000500"
flow_file1 = "work/inference/run.epoch-0-flow-field_pretrained/" + file_name + ".flo"
flow_file2 = "work/inference/run.epoch-0-flow-field_selftrained/" + file_name + ".flo"
cvb.show_flow(flow_file1)
cvb.show_flow(flow_file2)