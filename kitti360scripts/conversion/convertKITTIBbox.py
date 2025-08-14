from kitti360scripts.helpers.annotation import Annotation3D
import os
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import cv2
import glob
import argparse
from tqdm import tqdm

# python convertKITTIBbox.py --datadir /data1/datasets/KITTI-360 --seq 2013_05_28_drive_0000_sync --out ./kitti360_bbox
def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--seq', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--range', type=int, default=100)
    parser.add_argument('--cls', type=str, default='car')
    parser.add_argument('--vis', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    datadir = args.datadir
    seq = args.seq
    out = args.out
    max_dist = args.range
    cls = args.cls
    visualize = args.vis

    ann_3d_dir = os.path.join(datadir, 'data_3d_bboxes')
    annotation3D = Annotation3D(ann_3d_dir, seq)

    # Cam_00 intrinsic
    intrinstic_file = os.path.join(datadir, 'calibration', 'perspective.txt')
    if not os.path.exists(intrinstic_file):
        raise FileNotFoundError(f"Error: Intrinsic file not found at {intrinstic_file}")
    with open(intrinstic_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = line.strip().split()
            if lineData[0] == 'P_rect_00:':
                K_00 = np.array(lineData[1:]).reshape(3,4).astype(np.float64)
                K_00 = K_00[:,:-1]

    # Cam_00 extrinsic
    CamPose_00 = {}
    extrinstic_file = os.path.join(datadir, 'data_poses', seq)
    cam2world_file_00 = os.path.join(extrinstic_file,'cam0_to_world.txt')
    if not os.path.exists(cam2world_file_00):
        raise FileNotFoundError(f"Error: Extrinsic file not found at {cam2world_file_00}")
    with open(cam2world_file_00,'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = list(map(float,line.strip().split()))
            CamPose_00[int(lineData[0])] = np.array(lineData[1:]).reshape(4,4)

    K = K_00
    
    images = sorted(glob.glob(os.path.join(datadir, 'data_2d_raw', seq, 'image_00/data_rect', "*.png")))
    if len(images)==0:
        raise FileNotFoundError(f"Error: Images not found at {os.path.join(datadir, seq, 'image_00/data_rect')}")
        
    os.makedirs(out, exist_ok=True)
    cano_verts = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5,  0.5], [0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5]])
    im_width=None
    im_height=None

    with open(os.path.join(out, 'kitti_style.txt'), 'w') as f:
        for idx, im_path in tqdm(enumerate(images)):
            frame = int(os.path.basename(im_path).split('.')[0])
            if frame not in CamPose_00:
                continue
            c2w = CamPose_00[frame]

            if (visualize and idx % 50 == 0) or im_width is None:
                im = cv2.imread(im_path)
                im_width = im.shape[1]
                im_height = im.shape[0]
                
            for iid, annotation in annotation3D.objects.items():
                timestamps = list(annotation.keys())
                # static object
                if len(timestamps)==1 and timestamps[0]==-1:
                    obj = annotation[timestamps[0]]
                # dynamic object
                else:
                    if frame in timestamps:
                        obj = annotation[frame]
                    else:
                        continue

                if obj.name != cls:
                    continue
                R = obj.R
                T = obj.T
                if np.linalg.norm(T - c2w[:3, 3]) > max_dist:
                    continue
                bsize = np.linalg.norm(R, axis=0) # l, w, h
                R = R / bsize
                b2w = np.eye(4)
                b2w[:3, :3] = R
                b2w[:3, 3] = T
                c2b = np.linalg.inv(b2w) @ c2w
                b2c = np.linalg.inv(c2b)


                # === yaw angle ===
                b2c_kitti360=b2c[:3,:3]
                # remap: KITTI box axes = [x_right, y_down, z_forward] = [-y_box, -z_box, x_box]
                b2c_kitti = b2c_kitti360 @ np.array([[0,0,1],
                                                     [-1,0,0],
                                                     [0,-1,0]])
                
                rotation_y = np.arctan2(b2c_kitti[0, 2], b2c_kitti[2, 2])
                yaw=rotation_y
                # use the yaw angle to calcuate the "de-generated" rotation matrix
                b2c[:3, :3] = SCR.from_euler('y', yaw).as_matrix()


                # === 2D bounding box ===
                # project vertices to the 2D space 
                # bsize = [l, w, h] in KITTI-360 axes (x_fwd, y_left, z_up)
                # remap: KITTI box axes = [x_right, y_down, z_forward] = [-y_box, -z_box, x_box]
                verts = cano_verts * bsize[[1, 2, 0]]  # reorders dimensions to match remapped axes

                verts_cam = (b2c[:3, :3] @ verts.T).T + b2c[:3, 3]
                verts_uvz = (K[:3, :3] @ verts_cam.T).T
                verts_uv = (verts_uvz[:, :2] / verts_uvz[:, 2][:, None]).astype(int)
                mask = (verts_uvz[:, 2] > 0) & (verts_uv[:, 0] >= 0) & (verts_uv[:, 1] >= 0) & (verts_uv[:, 1] < im_height) & (verts_uv[:, 0] < im_width)
                if mask.sum() < 4:
                    continue

                xmin = np.min(verts_uv[:, 0])
                ymin = np.min(verts_uv[:, 1])
                xmax = np.max(verts_uv[:, 0])
                ymax = np.max(verts_uv[:, 1])
                
                # clamp to image bounds
                xmin_clamped = np.clip(xmin, 0, im_width - 1)
                ymin_clamped = np.clip(ymin, 0, im_height - 1)
                xmax_clamped = np.clip(xmax, 0, im_width - 1)
                ymax_clamped = np.clip(ymax, 0, im_height - 1)

                # === occlusion ===
                occlusion = 3 # unknown, this could be calculated by determining the occlusion relationship of multiple bounding boxes
  

                # === truncation ===
                truncation = 1 -  ((ymax_clamped-ymin_clamped)*(xmax_clamped-xmin_clamped))/((ymax-ymin)*(xmax-xmin))


                # === alpha ===
                alpha = yaw - np.arctan2(b2c[0,3], b2c[2,3])
                # normalize alpha to [-pi, pi]
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                elif alpha < -np.pi:
                    alpha += 2 * np.pi


                # frameid, yaw, x, y, z, l, h, w
                pos = ' '.join([str(p) for p in b2c[:3, 3].tolist()])
                #lhw = ' '.join([str(s) for s in bsize[[0, 2, 1]].tolist()])
                hwl = ' '.join([str(s) for s in bsize[[2, 1, 0]].tolist()])
                bbox_2d = ' '.join([str(s) for s in [xmin_clamped, ymin_clamped, xmax_clamped, ymax_clamped]])
                f.write(f'{frame} {obj.instanceId} {truncation} {occlusion} {alpha} {bbox_2d} {hwl} {pos} {rotation_y}\n')

                if visualize and idx % 50 == 0:
                    connections = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                                    [1, 4], [0, 5], [3, 6], [2, 7]]
                    for connection in connections:
                        line = verts_uv[connection, :].tolist()
                        cv2.line(im, line[0], line[1], color=(0,0,255), thickness=1)

                    # Compute a base position for the text (center of bbox top)
                    base_pos = ( (verts_uv[0] + verts_uv[5]) // 2 )  # (x, y)
                    x_, y_ = int(base_pos[0]), int(base_pos[1])
                    line_height = 15  # pixels between lines
                    cv2.putText(im, str(obj.instanceId), (x_, y_ - line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(im, f'alpha: {np.degrees(alpha):.2f}', (x_, y_), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(im, f'rot_y: {np.degrees(yaw):.2f}', (x_, y_ + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    # Draw local coordinate axes
                    axis_len = 1.5  # meters
                    origin_cam = b2c[:3, 3]
                    x_axis_cam = origin_cam + b2c[:3, 0] * axis_len
                    y_axis_cam = origin_cam + b2c[:3, 1] * axis_len
                    z_axis_cam = origin_cam + b2c[:3, 2] * axis_len

                    def project_point(pt_cam):
                        uvz = K @ pt_cam
                        return (int(uvz[0] / uvz[2]), int(uvz[1] / uvz[2]))

                    origin_uv = project_point(origin_cam)
                    x_axis_uv = project_point(x_axis_cam)
                    y_axis_uv = project_point(y_axis_cam)
                    z_axis_uv = project_point(z_axis_cam)

                    # Red = x, Green = y, Blue = z
                    cv2.line(im, origin_uv, x_axis_uv, (0, 0, 255), 2)
                    cv2.line(im, origin_uv, y_axis_uv, (0, 255, 0), 2)
                    cv2.line(im, origin_uv, z_axis_uv, (255, 0, 0), 2)


                    # Draw 2D bounding box 
                    pt1 = (int(xmin), int(ymin))  # top-left corner
                    pt2 = (int(xmax), int(ymax))  # bottom-right corner
                    
                    color = (0, 120, 255)
                    
                    # Thickness of the rectangle border
                    thickness = 1
                    
                    # Draw rectangle on the image
                    cv2.rectangle(im, pt1, pt2, color, thickness)


            if visualize and idx % 50 == 0:
                out_path = os.path.join(out, f"{str(frame).zfill(10)}_orig.png")
                cv2.imwrite(out_path, im)

