#import matplotlib
#matplotlib.use('Agg')
from utils.flowlib import flow_to_image
import matplotlib.pyplot as plt

import cvbase as cvb 
import numpy as np 
from utils.flow_utils import readFlow
import cv2
import os


pretrained_folder = "work/inference/run.epoch-0-flow-field_pretrained/"
selftrained_folder = "work/inference/run.epoch-0-flow-field/"

combined_folder = "work/inference/combined/"
if not os.path.exists(combined_folder):
	os.makedirs(combined_folder)

#file_name = "000500"
'''
for file_name in os.listdir(pretrained_folder):
	pretrained_flow_file = pretrained_folder + file_name
	selftrained_flow_file = selftrained_folder + file_name
	#cvb.show_flow(pretrained_flow_file)
	#pretrained_flow = flow_to_image(pretrained_flow_file)
	#pretrained_flow = pretrained_flow[0].numpy().transpose((1,2,0))
	#
	pretrained_flow = readFlow(pretrained_flow_file)
	pretrained_im = flow_to_image(pretrained_flow)

	selftrained_flow = readFlow(selftrained_flow_file)
	selftrained_im = flow_to_image(selftrained_flow)

	#plt.imshow(pretrained_im)
	#plt.savefig(file_name + '_pretrained.png', bbox_inches='tight')

	#plt.imshow(selftrained_im)
	#plt.savefig(file_name + '_selftrained.png', bbox_inches='tight')

	combined = np.concatenate((pretrained_im, selftrained_im), axis=1)
	cv2.imwrite(combined_folder + file_name.split('.')[0] + '_predicted.png', combined)
'''

img_list = os.listdir(combined_folder)
img_list.sort()
print(img_list)

frame = cv2.imread(combined_folder + '/' + img_list[0])
height, width, _ = frame.shape

output = "work/inference/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 10.0, (width, height))

for image_name in img_list:
	image_path = os.path.join(combined_folder, image_name)
	frame = cv2.imread(image_path)

	out.write(frame)
	cv2.imshow('video', frame)
	if (cv2.waitKey(1) & 0xFF) == ord('q'): break

out.release()
cv2.destroyAllWindows()
