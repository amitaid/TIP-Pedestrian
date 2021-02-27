from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np

import heurisitics

import math
from sklearn.neighbors import KNeighborsClassifier

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
from pathlib import Path
from pandas import read_csv

NECK = 0
NOSE = 1
PELVIS = 2 
L_SHO = 3
L_ELB = 4
L_WRI = 5
L_HIP = 6
L_KNEE = 7
L_ANK = 8
R_SHO = 9
R_ELB = 10
R_WRI = 11
R_HIP = 12
R_KNEE = 13
R_ANK = 14
R_EYE = 15
L_EYE = 16
R_EAR = 17
L_EAR = 18

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d


#helper functions for heurisitics.

def check_phone_left(pose,canvas_3d):
    #If distance between wrist and ear smaller than distance between should and ear 
    l_ear_to_wrist_distance = math.sqrt(pow(pose[L_WRI][0]-pose[L_EAR][0],2)+pow(pose[L_WRI][1]-pose[L_EAR][1],2)+pow(pose[L_WRI][2]-pose[L_EAR][2],2))
    #print(l_ear_to_wrist_distance)
    if  50 > l_ear_to_wrist_distance and pose[L_WRI][2]>pose[L_SHO][2]:
        cv2.putText(canvas_3d, "Talking on phone" ,
            (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        # print("A leftie is talking on the phone")
    #print(pose)
def avg_2_pose(pose1, pose2):
    return [(pose1[0] + pose2[0]) / 2, (pose1[1] + pose2[1]) / 2, (pose1[2] + pose2[2]) / 2]

def check_lying_down(pose, canvas_3d):
    # distance of ankles to head in z axis is low 
    ankle_median_pos = avg_2_pose(pose[L_ANK], pose[R_ANK])
    
    # print(pose[NECK][2] - ankle_median_pos[2])
    if pose[NECK][2] - ankle_median_pos[2] < 60 and pose[NECK][2] - ankle_median_pos[2] != 0 :
         cv2.putText(canvas_3d, "Lying down" ,
            (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

def check_standing(pose, canvas_3d):
    # distance of ankles to head in z axis is high 
    ankle_median_pos = avg_2_pose(pose[L_ANK], pose[R_ANK])
    
    # print(pose[NECK][2] - ankle_median_pos[2])
    if pose[NECK][2] - ankle_median_pos[2] > 100 and pose[NECK][2] - ankle_median_pos[2] != 0 :
         return True; 

def check_waving_right(pose, canvas_3d):
    # print("right wrist z " , pose[R_WRI][2], "  ||  neck z ",pose[NECK][2])
    # print("offset   ",pose[R_WRI][2] - pose[NECK][2] ) 
    if pose[R_WRI][2] - pose[NECK][2]  > 20 and pose[R_WRI][2] != 0 and pose[NECK][2] != 0 and pose[R_WRI][2] != pose[NECK][2] :
        cv2.putText(canvas_3d, "Wave right hand",
        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
def center_mass(pose):
    x = pose[L_SHO][0] + pose[L_HIP][0] + pose[R_SHO][0] + pose[R_HIP][0]
    y = pose[L_SHO][1] + pose[L_HIP][1] + pose[R_SHO][1] + pose[R_HIP][1]
    z = pose[L_SHO][2] + pose[L_HIP][2] + pose[R_SHO][2] + pose[R_HIP][2]
    return [x / 4, y / 4, z / 4]
    
def check_moving(pose, canvas_3d, moving_list, idx, last_pose):
    center_curr = center_mass(pose)
    center_last = center_mass(last_pose)
    delta = abs(center_curr[0] - center_last[0]) + abs(center_curr[1] - center_curr[1])
    if delta < 100:
        moving_list[idx] = delta
    print(sum(moving_list[idx - 20: idx]))
    if sum(moving_list[idx - 20:idx]) > 20:
        return True
    return False
    # look at hip(center mass), if is moving more the delta in X last frames, declare moving 
    
def parse_args():
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, default="human-pose-estimation-3d.pth")
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='sample4.mp4')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--use-tensorrt', help='Optional. Run network with TensorRT as inference engine.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    stride = 8
 
    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(args.model, args.device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(args.model, args.device, use_tensorrt=args.use_tensorrt)

    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    
    # Load dataset
    path = "3d\\Poses\\posesData.csv"     # url or just path
    dataset = read_csv(path)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:57]
    y = array[:, 57]
   
    # load model for pose classification .
  
    model = KNeighborsClassifier()                  # using now the KNN model
    model.fit(X, y)                                 # fitting (learning) the training set


    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    limitter  = 0
    moving_list = [0] * 1000 * 2 
    idx = 0


    last_pose = np.empty(shape=(57,1))
    for frame in frame_provider:
   
        current_time = cv2.getTickCount()
        if frame is None:
            break
      
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        edges = []
        
        predictions = []
        if len(poses_3d):
            #Code for rotation and drawing.
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]   
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
    

        plotter.plot(canvas_3d, poses_3d, edges)
        

        for pose in poses_3d:
            ## ML classification.
            X = pose.flatten()
            prediction = model.predict(X.reshape(1,-1)) #reshape to annouce a single sample 
            print(prediction)
            cv2.putText(frame, prediction[0],
                (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

                ## heuristic classification ##
            
            check_lying_down(pose,canvas_3d) 
            check_phone_left(pose, canvas_3d)         
            check_waving_right(pose,canvas_3d)
            standing_flag = check_standing(pose,canvas_3d)
            idx = idx + 1

            moving_flag = False
            if idx >= 30:
                check_moving(pose, canvas_3d, moving_list, idx ,last_pose)
                
            if moving_flag:
                cv2.putText(canvas_3d, "Walking",(40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            else:
                if standing_flag:
                    cv2.putText(canvas_3d, "Standing" ,
                    (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                    last_pose = pose 
    
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        draw_poses(frame, poses_2d)
       
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        
        cv2.imshow('ICV 3D Human Pose Estimation', frame)

        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1
   
