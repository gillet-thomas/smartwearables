# Copyright (c) Facebook, Inc. and its affiliates.

import os.path as osp
import torch
import numpy as np
import cv2

from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
from mocap_utils.timer import Timer

from renderer.visualizer import Visualizer

from VirtualDresser import embed_colab
import glob, time, random


def getBackground(rand = False):
    if rand:
        backgrounds = glob.glob("data/backgrounds/*.jpg")
        index = random.randrange(0, len(backgrounds))
        return cv2.imread(backgrounds[index])
    else:
        return np.zeros((2288, 1080, 3), np.uint8)

def clothesButton(frame, rWrist, lWrist, clothesTimer, duration):
    (btnX1, btnY1) = (15,15)
    (btnX2, btnY2) = (150,150)
    DURATION_MAX = 2

    changed = False
    rWristX = rWrist[0]
    rWristY = abs(rWrist[1])
    lWristX = lWrist[0]
    lWristY = abs(lWrist[1])

    # If one of the wrist is in the area, start the computation
    if (rWristX >= btnX1 and rWristX <= btnX2 and rWristY >= btnY1 and rWristY <= btnY2) or (lWristX >= btnX1 and lWristX <= btnX2 and abs(lWristY) >= btnY1 and abs(lWristY) <= btnY2):
        if clothesTimer is None:
            clothesTimer = time.time()
        else:
            duration = time.time() - clothesTimer
            progress = min((btnX2-btnX1)*(duration/DURATION_MAX), (btnX2-btnX1))
            cv2.rectangle(frame, (btnX1, btnY1), (int(btnX1+progress), btnY2), (0,0,255), thickness=-1)
            if duration > DURATION_MAX:
                clothesTimer = None
                duration = 0
                changed=True
    else:
        clothesTimer = None
        duration = 0

    cv2.rectangle(frame, (btnX1, btnY1), (btnX2, btnY2), (0,0,255), thickness=2)
    cv2.putText(frame, "Change", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), thickness=2)
    cv2.putText(frame, "clothes", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), thickness=2)

    return clothesTimer, duration, changed

def run_body_mocap(args, body_bbox_detector, body_mocap, visualizer):
    duration = 0
    clothesTimer = None

    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    timer = Timer()

    # Set background and clothes    
    background = getBackground(True)

    while True:
        timer.tic()

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None
        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
        elif input_type == 'webcam':    
            _, img_original_bgr = input_data.read()  #True and image is received
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   

        # Flip the input image
        img_original_bgr = cv2.flip(img_original_bgr, 1)

        # Resize the input image directly
        max_height = 720
        adjusted_width = int(img_original_bgr.shape[1]/(img_original_bgr.shape[0]/max_height))
        img_original_bgr = cv2.resize(img_original_bgr, (adjusted_width, max_height), interpolation=cv2.INTER_AREA)

        _, body_bbox_list = body_bbox_detector.detect_body_pose(img_original_bgr)
        if len(body_bbox_list) < 1: 
            print(f"No body detected: {image_path}")
            continue

        print("--------------------------------------")

        #Sort the bbox using bbox size (to make the order as consistent as possible without tracking)
        bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]

        # Body Pose Regression
        pred_output_list, joints = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualization
        resizedBg = cv2.resize(background, (img_original_bgr.shape[1], img_original_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        res_img = visualizer.visualize(resizedBg, pred_mesh_list = pred_mesh_list, body_bbox_list = body_bbox_list)

        # Display button to change clothes, and dress avatar
        clothesTimer, duration, newClothes = clothesButton(res_img, joints[31], joints[36], clothesTimer, duration)
        embed_colab.dress(res_img, joints, newClothes)
        cv2.imshow("final result", res_img)

        #Save the result
        demo_utils.save_res_img(args.out_dir, image_path, res_img)

        timer.toc(bPrint=True, title="Time")
        print(f"Processed : {image_path}")
        # cv2.waitKey(0)
        # break

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

def main():
    args = DemoOptions().parse()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator()

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    visualizer = Visualizer(args.renderer_type)
  
    run_body_mocap(args, body_bbox_detector, body_mocap, visualizer)


if __name__ == '__main__':
    main()