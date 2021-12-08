import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import pickle
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import json

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton

def read_wrnch(file_path):
    f = open(file_path,)
    data = json.load(f)
    return data

def get_body_pose(wrnch_data, frame_no):
    frame = wrnch_data["frames"][frame_no]
    if ("persons" not in frame or len(frame["persons"]) == 0):
        print("No person")
        return None

    return frame["persons"]

def get_bboxes(wrnch_data, frame_no, frame_shape):
    pose_2ds = get_body_pose(wrnch_data, frame_no)
    if (pose_2ds is None):
        return [None]

    bboxes = []
    for person in pose_2ds:
        pose2d = person["pose2d"]
        bboxes.append([
            pose2d["bbox"]["minX"] * frame.shape[1],
            pose2d["bbox"]["minY"] * frame.shape[0],
            pose2d["bbox"]["width"] * frame.shape[1],
            pose2d["bbox"]["height"] * frame.shape[0],
            person["id"]
        ])

    return bboxes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--wrnch_path', type=str, dest='wrnch_path')
    parser.add_argument('--video_path', type=str, dest='video_path')
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def keypoints_to_hrnet(keypoints_2d):
    conversion_order = [9, 0, 0, 0, 0, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    reordered = keypoints_2d[conversion_order]
    confidences = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return np.hstack((reordered, confidences[:, None]))

def bbox_to_hrnet(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

if __name__ == "__main__":
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    # MuCo joint set
    joint_num = 18
    joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax') 
    flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

    # snapshot load
    model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_pose_net(cfg, False, joint_num)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()

    # prepare input image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    cap = cv2.VideoCapture(args.video_path)
    wrnch_data = read_wrnch(args.wrnch_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    writer = cv2.VideoWriter(
        filename=args.output_path,
        fps=30,
        fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        frameSize=(width, height)
    )
    
    writer_3d = cv2.VideoWriter(
        filename=args.output_path.split(".")[0]+"_3d.mp4",
        fps=30,
        fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        frameSize=(640, 480)
    )

    frame_no = 0
    pose_results_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_img = frame
        original_img_height, original_img_width = original_img.shape[:2]
        pose_result = []
        bbox_list = get_bboxes(wrnch_data, frame_no, frame.shape)
        if (bbox_list[0] is None):
             writer.write(frame)
             frame_no += 1
             pose_results_list.append(pose_result)
             continue
        root_depth_list = [11250.5732421875]

        person_num = len(bbox_list)

        # normalized camera intrinsics
        focal = [1500, 1500] # x-axis, y-axis
        princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis

        # for each cropped and resized human image, forward it to PoseNet
        output_pose_2d_list = []
        output_pose_3d_list = []
        
        for n in range(person_num):
            bbox_data = bbox_list[n][:4]
            person_id = bbox_list[n][4]
            bbox = process_bbox(np.array(bbox_data), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
            img = transform(img).cuda()[None,:,:,:]

            # forward
            with torch.no_grad():
                pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

            # inverse affine transform (restore the crop and resize)
            pose_3d = pose_3d[0].cpu().numpy()
            pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
            pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            pose_2d_coords = pose_3d[:,:2].copy()
            output_pose_2d_list.append(pose_2d_coords)
            
            # root-relative discretized depth -> absolute continuous depth
            pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
            pose_3d = pixel2cam(pose_3d, focal, princpt)
            output_pose_3d_list.append(pose_3d.copy())
            pose_result.append({
                "bboxes": bbox_to_hrnet(bbox_list[n]),
                "keypoints": keypoints_to_hrnet(pose_2d_coords),
                "area": bbox_list[n][2] * bbox_list[n][3],
                "track_id": person_id
            })
        pose_results_list.append(pose_result)
        # visualize 2d poses
        vis_img = original_img.copy()
        for n in range(person_num):
            vis_kps = np.zeros((3,joint_num))
            vis_kps[0,:] = output_pose_2d_list[n][:,0]
            vis_kps[1,:] = output_pose_2d_list[n][:,1]
            vis_kps[2,:] = 1
            vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
        writer.write(vis_img)

        # visualize 3d poses
        #vis_kps = np.array(output_pose_3d_list)
        #output_viz = vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')
        #writer_3d.write(output_viz)
        print(f"Processing frame {frame_no} of {len(wrnch_data['frames'])}")
        frame_no += 1
    with open(f'{args.output_path.split(".")[0]}.p', 'wb') as outfile:
        pickle.dump(pose_results_list, outfile)
    writer.release()
    writer_3d.release()
    cap.release()
