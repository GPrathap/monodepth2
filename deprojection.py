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
from collections import defaultdict
import ros_numpy
from random import randrange

buffer = collections.deque(maxlen=5)

focal_length_left = 800
focal_length_right = 800

is_set_camera_info=False

class PoseEstimator:
    def __init__(self):
        self.image_path = ""
        self.left_info_K = None 
        self.left_info_D = None 
        self.depth_map = None
        self.R = None
        self.T = None
        self.R_vector = None
        self.dempth_mapper = defaultdict(list)
        self.allowed_off_set = 1.5
        self.homograpy_estimated = False

    def handle_odometry(self, msg):
        mess = msg
    
    def camera_info_callback(self, info_left):
        self.K = np.array(info_left.K).reshape([3, 3])
        self.D = np.array(info_left.D)
        # print("--------------------")
        # print(info_left.R)
        self.R = np.array(info_left.R).reshape([3, 3])
        self.R = np.eye(3)
        self.T = np.array(info_left.P).reshape([3, 4])[:,3]
        self.R_vector, _ = cv2.Rodrigues(self.R)
        # print("================================||||||||")
        # print(self.R_vector)
    
    def point_cloud_callback(self, point_cloud_left):
        # print("---------")
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(point_cloud_left)
        self.depth_map = xyz_array
        # print(self.depth_map.shape)

    def callback(self, rgb_left):
        left_image = CvBridge().imgmsg_to_cv2(rgb_left, desired_encoding="bgr8")
        # # print(left_image.shape)
        # # frame = self.process(left_image)
        # frame = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        # img = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        # num_labels, labels = cv2.connectedComponents(img)

        # label_hue = np.uint8(179*labels/np.max(labels))
        # blank_ch = 255*np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # # cvt to BGR for display
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # blank_image = np.zeros((label_hue.shape[0], label_hue.shape[1], 3), dtype=np.uint8)
        # # set bg label to black
        # # ffg = label_hue==0
        # # print(ffg[0])
        # # blank_image[label_hue==0] = 240
        # # blank_image[label_hue!=0] = 0
        # # blank_image.fill(120)
        # counter = 0
        # for k in range(0, label_hue.shape[0]):
        #     for l in range(0, label_hue.shape[1]):
        #         if(label_hue[k,l] == 0):
        #             blank_image[k,l].fill(255)
        #             counter +=1

        # msg_frame = CvBridge().cv2_to_imgmsg(blank_image, encoding="bgr8")
        # depth_img_left.publish(msg_frame)

        # r, c = np.where(label_hue==0)
        # check_pose = []
        # for i, j in zip(r, c):
        #     check_pose.append([i, j])
        
        # blank_image = np.zeros((label_hue.shape[0], label_hue.shape[1], 3), dtype=np.uint8)
       
        # check_pose = np.array(check_pose)
       
        # for index in check_pose:
        #     blank_image[index[0], index[1]] = 255
        # msg_frame = CvBridge().cv2_to_imgmsg(blank_image, encoding="rgb8")
        # depth_img_projection.publish(msg_frame)

        
                           
        rows_color = left_image.shape[0]
        # print(imgpts.shape)
        if(self.R_vector is not None):
            # self.depth_map = np.array(self.depth_map, dtype=np.float)
            if(self.depth_map is not None):
                imgpts, _ = cv2.projectPoints(self.depth_map, self.R_vector, self.T, self.K, self.D)
                imgpts = np.int32(imgpts).reshape(-1, 2)
                min_val = min(imgpts.shape[0], self.depth_map.shape[0])-20
                points_3D = []
                points_2D = []
                for g in range(0, 50):
                    index = randrange(min_val)
                    points_3D.append(self.depth_map[index])
                    points_2D.append(imgpts[index])
                
               
                points_3D = np.array(points_3D,  dtype=np.float)
                points_2D = np.array(points_2D,  dtype=np.float)
                print(points_3D.shape)
                print(points_2D.shape)
                
                _, R_exp, tvec = cv2.solvePnP(points_3D,
                              points_2D, self.K, self.D)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)
                
                rotationMatrix, _ = cv2.Rodrigues(R_exp)
                tvec = np.squeeze(tvec)

                for i in range(0, 20):
                    uvPoint = np.array([points_2D[i][0], points_2D[i][1], 1])
                    world_point = points_3D[i]
                    rot_inv = np.linalg.inv(rotationMatrix)
                    cam_inv = np.linalg.inv(self.K)
                    leftSideMat = np.dot(rot_inv, np.dot(cam_inv, uvPoint))
                    rightSideMat = np.dot(rot_inv, tvec)
                    s = (world_point[2] + rightSideMat[2])/leftSideMat[2]
                    # print((s*np.dot(cam_inv,uvPoint)).shape)
                    # print(tvec.shape)
                    # print((s*np.dot(cam_inv,uvPoint) - tvec).shape)
                    Pose = np.dot(rot_inv , (s*np.dot(cam_inv,uvPoint) - tvec))

                    print("Real and estimated")
                    print(Pose)
                    print(world_point)

                # print("=====================")
                # print(label_hue.shape)
                # print(left_image.shape)
        #         self.dempth_mapper = defaultdict(list)
        #         
        #         for i in range(0, min_val):
        #             index = imgpts[i][0]*rows_color  + imgpts[i][1]
        #             depth_val = self.depth_map[i]
        #             # print(imgpts[i][0])
        #             # print(imgpts[i][1])
        #             # print("-----")
        #             self.dempth_mapper[index].append(depth_val)

        #         interested_points = []
        #         r, c = np.where(label_hue==0)
        #         check_pose = []
        # #         # print(self.dempth_mapper)
        #         for i, j in zip(r, c):
        # #             # print(i,j)
        #             index = i*rows_color+j
        #             if(len(self.dempth_mapper[index]) >0 ):
        #                 # print(self.dempth_mapper[index])
        #                 for pose in self.dempth_mapper[index]:
        #                     pose = pose.tolist()
        #                     pose.append(i)
        #                     pose.append(j)
        #                     check_pose.append([i, j])
        #                     # check_pose.append(j)
        #                     interested_points.append(pose)
                
        #         interested_points = np.array(interested_points)
        #         # interested_points = interested_points.flatten()   
        #         # print("=================depth map================================")
        #         # print(interested_points.shape)
        #         # print(interested_points[0][0])
        #         # for i in range(0, interested_points.shape[0]):
        #         #     for 
        #         filtered_points = None
        #         if(interested_points.shape[0]>1):
        #             # print(interested_points[:,0].max())
        #             # print(interested_points[:,1].max())
        #             # print(interested_points[:,2].max())
        #             filtered_points = np.where(interested_points[:,2] <= interested_points[:,2].max()-self.allowed_off_set)
        #             filtered_points = interested_points[filtered_points][:,3:5]
        #             # filtered_points = interested_points[:,3:5]
        #         if(filtered_points is not None):
        #             # print(filtered_points.shape)
        #             # print(filtered_points[:,1].max())
        #             blank_image = np.zeros((label_hue.shape[0], label_hue.shape[1], 3), dtype=np.uint8)
        #             # print("============================")
        #             # print(blank_image.shape)
        #             # print(filtered_points[:,1].max())
        #             # print(filtered_points.shape)
        #             # print("Camera parameters...")
        #             # print(self.K)
        #             # print(self.R)
        #             # print(self.T)
        #             # print(self.R_vector)
        #             filtered_points = filtered_points.astype(np.int)
        #             # ank_image[label_hue==0] = 255
        #             # print(filtered_points[0:5])
        #             check_pose = np.array(check_pose)
        #             # print("size===")
        #             # print(check_pose.shape)
        #             # print(counter)
        #             # blank_image[filtered_points] = 255
        #             # print(check_pose[:,0].max())
        #             # print(check_pose[:,1].max())
        #             # check_pose = np.array([[3,496], [34,67], [56,78]])
        #             # print(check_pose)
        #             # for index in check_pose:
        #             #     blank_image[index[0], index[1]] = 255
        #             for index in filtered_points:
        #                 blank_image[index[0], index[1]] = 255
                    
        #             msg_frame = CvBridge().cv2_to_imgmsg(blank_image, encoding="rgb8")
        #             depth_img_projection.publish(msg_frame)



    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        #channel_count = img.shape[2]
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def drow_the_lines(self, img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # print(lines)
        if(lines is not None):
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img
    
    def process(self, image):
        # print(image.shape)
        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [
            (0, height),
            (width/2, height/2),
            (width, height)
        ]
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny_image = cv2.Canny(gray_image, 0, 80)
        cropped_image = self.region_of_interest(canny_image,
                        np.array([region_of_interest_vertices], np.int32),)
        lines = cv2.HoughLinesP(cropped_image,
                                rho=2,
                                theta=np.pi/180,
                                threshold=20,
                                lines=np.array([]),
                                minLineLength=10,
                                maxLineGap=200)
        image_with_lines = self.drow_the_lines(image, lines)
        return image_with_lines



if __name__ == '__main__':
    
    rospy.init_node('my_node', anonymous=True)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    point_cloud_estimator = PoseEstimator()

    # image_sub_left = message_filters.Subscriber('/inno_drone/down_camera/image_raw', Image)
    # info_sub_left = message_filters.Subscriber('/r200/depth/camera_info', CameraInfo)
    # depth_sub_left = message_filters.Subscriber('/r200/depth/points', PointCloud2)
    # point_cloud_left = rospy.Publisher('/depth/point_cloud_left', PointCloud2, queue_size=10)
    # point_cloud_right = rospy.Publisher('/depth/point_cloud_right', PointCloud2, queue_size=10)
    # ts = message_filters.ApproximateTimeSynchronizer([image_sub_left, info_sub_left, depth_sub_left], 10, 0.2)
    # ts.registerCallback(point_cloud_estimator.callback)
    # print(frame.shape)
    rospy.Subscriber('/realsense_camera/color/image_raw', Image, point_cloud_estimator.callback)
    rospy.Subscriber('/depth_registered/camera_info', CameraInfo, point_cloud_estimator.camera_info_callback)
    rospy.Subscriber('/depth_registered/points', PointCloud2, point_cloud_estimator.point_cloud_callback)
    # depth_img_right = rospy.Publisher('/depth/img_right', Image, queue_size=10)
    depth_img_left = rospy.Publisher('/depth/img_left', Image, queue_size=10)
    depth_img_projection = rospy.Publisher('/depth/img_left_deptch_projection', Image, queue_size=10)

    # rospy.Subscriber('/kitti/oxts/imu/', Imu, point_cloud_estimator.handle_odometry)


    rospy.spin()