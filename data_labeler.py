import message_filters
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import dlib
import argparse as ap
import collections
import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tqdm import tqdm
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

import argparse
import tf
import tf2_ros
import tf2_py as tf2
import geometry_msgs
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets
import networks
from layers import disp_to_depth, disp_to_depth_scaled
from utils import download_model_if_doesnt_exist

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2

import rospy
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
import std_msgs.msg

import ros_numpy
from saliency import BackPropagation
import uuid


buffer = collections.deque(maxlen=5)

focal_length_left = 800
focal_length_right = 800

is_set_camera_info=False


class DataLabeler:
    def __init__(self, labeled_data_dir):
        model_name = "mono+stereo_640x192"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.image_path = ""

        # download_model_if_doesnt_exist(args.model_name)
        model_path = os.path.join("models", model_name)
        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        print("Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4), is_scaled=True)

        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()
        
        self.labeled_data_dir = labeled_data_dir

    def depth_map_estimation(self, input_image, is_scaled):
        # with torch.no_grad():
            # Load image and preprocess
            # input_image = pil.open(image_path).convert('RGB')
        image_id = str(uuid.uuid4())[0:20]
        input_image = pil.fromarray(input_image)
        input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
        
        cv2.imwrite(self.labeled_data_dir + 'img/'+ image_id +'.png', cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR))    

        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(self.device)
        features = self.encoder(input_image)

        outputs = self.depth_decoder(features)
        
        disp = outputs
        disp_resized_np = disp.squeeze().cpu().detach().numpy()
        color_depth = self.get_color_image(disp_resized_np)
        cv2.imwrite(self.labeled_data_dir + 'depth/'+ image_id +'.png', cv2.cvtColor(np.array(color_depth), cv2.COLOR_RGB2BGR))
       
        disp_resized_np_low = np.where(disp_resized_np > disp_resized_np.max()/8.0, disp_resized_np, 0)
        disp_resized_np_high = np.where(disp_resized_np < disp_resized_np.max()/8.0, disp_resized_np, 0)
        disp_resized_np_high_save = np.where(disp_resized_np < disp_resized_np.max()/8.0, 255, 0)
        cv2.imwrite(self.labeled_data_dir + 'mask/'+ image_id +'.png', disp_resized_np_high_save)
        
        return self.get_color_image(disp_resized_np_low), disp_resized_np, self.get_color_image(disp_resized_np_high)

    def get_color_image(self, disp_resized_np):
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        # print('-> Done!')
        # im = pil.fromarray(colormapped_im)
        return colormapped_im

    def label_data(self, data_dir):
        for path, subdirs, files in os.walk(data_dir):
            for name in files:
                if(name.endswith(('.jpg', '.png'))):
                    print(os.path.join(path, name))
                    img = cv2.imread(os.path.join(path, name), 1)
                    self.depth_map_estimation(img, False)


if __name__ == '__main__':
    point_cloud_estimator = DataLabeler(labeled_data_dir="/home/geesara/dataset/")
    point_cloud_estimator.label_data("/home/geesara/dataset/data")
    rospy.spin()