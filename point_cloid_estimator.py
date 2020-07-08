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

from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

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

buffer = collections.deque(maxlen=5)

focal_length_left = 800
focal_length_right = 800

is_set_camera_info=False


class PointCloudEstimator:
    def __init__(self):
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
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()


    def depth_map_estimation(self, input_image, is_scaled):
        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            # Load image and preprocess
            # input_image = pil.open(image_path).convert('RGB')
            input_image = pil.fromarray(input_image)
            original_width, original_height = input_image.size
            input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(self.image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            if(is_scaled):
                scaled_disp, _ = disp_to_depth_scaled(disp, 0.1, 100, 4.4)
            else:
                scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = scaled_disp.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            print('-> Done!')
            im = pil.fromarray(colormapped_im)
            return colormapped_im, disp_resized_np

            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)

            # print("   Processed {:d} of {:d} images - saved prediction to {}".format(
            #     idx + 1, len(paths), name_dest_im))

            
    def xyz_array_to_pointcloud2(self, points, stamp=None, frame_id=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12*points.shape[0]
        msg.is_dense = int(np.isfinite(points).all())
        msg.data = np.asarray(points, np.float32).tostring()
        return msg 

    def handle_odometry(self, msg):
        mess = msg

    def callback(self, rgb_left, info_left, rgb_right, info_right):
        left_image = CvBridge().imgmsg_to_cv2(rgb_left, desired_encoding="rgb8")
        left_info_K = np.array(info_left.K).reshape([3, 3])
        left_info_D = np.array(info_left.D)
        # left_undist = cv2.undistort(left_image, left_info_K, left_info_D)
        
        left_depth,dis_left = self.depth_map_estimation(left_image, False)
        msg_frame = CvBridge().cv2_to_imgmsg(left_depth, encoding="rgb8")
        depth_img_left.publish(msg_frame)
       
        right_image = CvBridge().imgmsg_to_cv2(rgb_right, desired_encoding="rgb8")
        right_info_K = np.array(info_right.K).reshape([3, 3])
        right_info_D = np.array(info_right.D)

        right_depth, dis_right = self.depth_map_estimation(left_image, True)
        msg_frame = CvBridge().cv2_to_imgmsg(right_depth, encoding="rgb8")
        depth_img_right.publish(msg_frame)
        # right_undist = cv2.undistort(right_image, right_info_K, right_info_D)

        R = np.array(info_left.R).reshape([3, 3])
        T = np.array(info_left.P).reshape([3, 4])[:,3]
       
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_info_K, left_info_D,
            right_info_K, right_info_D,
            left_image.shape[:2],
            R,
            T, alpha=1.0)
        # print(dis_right.shape)
        # dis_img = cv2.cvtColor(dis_left, cv2.COLOR_RGB2BGR)
        # print(dis_left)
        points_3D = cv2.reprojectImageTo3D(dis_left, Q)
        # print(points_3D)
        points_3D = np.array(points_3D)/10000.0

        print("Message")
        try:
            transform = tf_buffer.lookup_transform("velo_link","camera_color_left", #source frame
                                    rospy.Time(0), #get the tf at first available time
                                    rospy.Duration(1.0))
            points = self.xyz_array_to_pointcloud2(points_3D, frame_id="velo_link", stamp=rospy.Time.now())
           
            # quad = transform.transform.rotation
            # print("before: ", quad)
            # orientation_list = [quad.x, quad.y, quad.z, quad.w]
            # (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
            (roll, pitch, yaw) = ((120*(np.pi/180)), (0*(np.pi/180)), (90*(np.pi/180)))
            # print (roll*(180.0/np.pi), pitch*(180.0/np.pi), yaw*(180.0/np.pi))
            quat = quaternion_from_euler (roll, pitch,yaw)
            # print("after: ", quat)
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]
            # transform = tf_buffer.lookup_transform("world","velo_link", #source frame
            #                         rospy.Time(0), #get the tf at first available time
            #                         rospy.Duration(1.0))

            # cloud_out = do_transform_cloud(cloud_out, transform)
             # print(transform.transform.rotation)
            cloud_out = do_transform_cloud(points, transform)
            point_cloud_left.publish(cloud_out)

            # points_3D = cv2.reprojectImageTo3D(dis_right, Q)
            # # print(points_3D)
            # points_3D = np.array(points_3D)/10000.0

            # points = self.xyz_array_to_pointcloud2(points_3D, frame_id="world", stamp=rospy.Time.now())
            # point_cloud_right.publish(points)
        except:
            print ("error")
            pass
        print("Transformed")

       
        
        # mapx1, mapy1 = cv2.initUndistortRectifyMap(left_info_K, left_info_D, R1, left_info_K,
        #                                         left_image.shape[:2],
        #                                         cv2.CV_32F)
        # mapx2, mapy2 = cv2.initUndistortRectifyMap(right_info_K, right_info_D, R1, right_info_K,
        #                                         right_image.shape[:2],
        #                                         cv2.CV_32F)
        
        # left_image = cv2.remap(left_image, mapx1, mapy1, cv2.INTER_LINEAR)
        # right_image = cv2.remap(right_image, mapx2, mapy2, cv2.INTER_LINEAR)

        # global is_set_camera_info
        # if(~is_set_camera_info):
        #     focal_length_left = left_info_K[0,0]
        #     folal_length_right = right_info_K[0,0]
        #     is_set_camera_info = True

        # img_pair = (left_image, right_image)
        # buffer.appendleft(img_pair)


if __name__ == '__main__':
    
    rospy.init_node('my_node', anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    point_cloud_estimator = PointCloudEstimator()
    
    image_sub_left = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
    info_sub_left = message_filters.Subscriber('/kitti/camera_color_left/camera_info', CameraInfo)
    image_sub_right = message_filters.Subscriber('/kitti/camera_color_right/image_raw', Image)
    info_sub_right = message_filters.Subscriber('/kitti/camera_color_right/camera_info', CameraInfo)
    point_cloud_left = rospy.Publisher('/depth/point_cloud_left', PointCloud2, queue_size=10)
    point_cloud_right = rospy.Publisher('/depth/point_cloud_right', PointCloud2, queue_size=10)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub_left, info_sub_left, image_sub_right, info_sub_right], 10, 0.2)
    ts.registerCallback(point_cloud_estimator.callback)
    depth_img_right = rospy.Publisher('/depth/img_right', Image, queue_size=10)
    depth_img_left = rospy.Publisher('/depth/img_left', Image, queue_size=10)

    rospy.Subscriber('/kitti/oxts/imu/', Imu, point_cloud_estimator.handle_odometry)


    rospy.spin()