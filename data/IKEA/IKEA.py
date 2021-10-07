import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from config import cfg
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, rigid_align, process_bbox
import cv2
import random
import json
from utils.vis import vis_keypoints, vis_3d_skeleton

class IKEA:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = "/home/fsuser/ProcessedDatasets/ANU_ikea_dataset_used_frames"
        self.annot_path = "/home/fsuser/ProcessedDatasets/ANU_ikea_dataset_used_frames"
        self.train_annot_path = "/home/fsuser/ProcessedDatasets/ANU_ikea_dataset_used_frames/ikea_training.json"
        self.human_bbox_root_dir = osp.join('..', 'data', 'Human36M', 'bbox_root', 'bbox_root_human36m_output.json')
        self.joint_num = 18 # Orig 17 (as per COCO, added pelvis)

        self.joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', \
            'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', \
            'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', \
            'R_Knee', 'L_Ankle', 'R_Ankle', "Pelvis"
        ) # Same as MSCOCO, added pelvis

        self.lhip_idx = self.joints_name.index('L_Hip')
        self.rhip_idx = self.joints_name.index('R_Hip')

        self.flip_pairs = ( )
        self.skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3),
            (6, 8), (8, 10), (5, 7), (7, 9),
            (12, 14), (14, 16), (11, 13), (13, 15),
            (5, 6), (11, 12)
        )
        self.joints_have_depth = True
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
        self.protocol = 2
        self.data = self.load_data()

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1,5,6,7,8,9]
            elif self.protocol == 2:
                subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def add_thorax(self, joint_coord):
        thorax = (joint_coord[self.lshoulder_idx, :] + joint_coord[self.rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(self.train_annot_path)
            data = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                width, height = img['width'], img['height']

                bbox = process_bbox(ann['bbox'], width, height) 
                if bbox is None: continue

                # joints and vis
                f = np.array(db.imgs[ann['image_id']]["camera_param"]['focal'])
                c = np.array(db.imgs[ann['image_id']]["camera_param"]['princpt'])

                # Ignore the 4th column (which is confidence I believe)
                joint_cam = np.array(ann['joint_cam'])[:, 0:3]
                pelvis = (joint_cam[self.lhip_idx] + joint_cam[self.rhip_idx])/2.0

                joint_cam = np.vstack((joint_cam, pelvis))
                joint_img = cam2pixel(joint_cam, f, c)
                joint_img[:,2] = joint_img[:,2] - joint_cam[self.root_idx,2]

                # If all of the coords of a specific joint are 0, it's invalid
                joint_vis = np.expand_dims(np.where(np.all(joint_cam == 0, axis=1), 0, 1), 1)
                joint_img[:,2] = 0
                
                img_path = osp.join(self.img_dir, db.imgs[ann['image_id']]['file_name'])
                data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                    'joint_vis': joint_vis,
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'f': f, 
                    'c': c 
                })
        return data

    def evaluate(self, preds, result_dir):
        
        print('Evaluation start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)
        
        pred_save = []
        error = np.zeros((sample_num, self.joint_num-1)) # joint error
        error_action = [ [] for _ in range(len(self.action_name)) ] # error for each sequence
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']
            gt_vis = gt['joint_vis']
            
            # restore coordinates to original space
            pred_2d_kpt = preds[n].copy()
            pred_2d_kpt[:,0] = pred_2d_kpt[:,0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_2d_kpt[:,1] = pred_2d_kpt[:,1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:,2] = (pred_2d_kpt[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + gt_3d_root[2]

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1,500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3,self.joint_num))
                tmpkps[0,:], tmpkps[1,:] = pred_2d_kpt[:,0], pred_2d_kpt[:,1]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)
 
            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[self.root_idx]
           
            if self.protocol == 1:
                # rigid alignment for PA MPJPE (protocol #1)
                pred_3d_kpt = rigid_align(pred_3d_kpt, gt_3d_kpt)
            
            # exclude thorax
            pred_3d_kpt = np.take(pred_3d_kpt, self.eval_joint, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.eval_joint, axis=0)
           
            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2,1))
            img_name = gt['img_path']
            action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': image_id, 'joint_cam': pred_3d_kpt.tolist(), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()}) # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' error (' + metric + ') >> tot: %.2f\n' % (tot_err)

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
           
        print(eval_summary)

        # prediction save
        output_path = osp.join(result_dir, 'bbox_root_pose_human36m_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

