import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd


def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
    axes.lines  = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
    axes.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0,1]])
    return axes


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3,1)], axis=1)
    return transform_mat


def make_unit_cube_face_points(res=40):
    u = np.linspace(0.0, 1.0, res)
    uu, vv = np.meshgrid(u, u)
    uu = uu.reshape(-1,1)
    vv = vv.reshape(-1,1)
    faces = []
    colors = []
    faces.append(np.hstack([uu, vv, np.zeros_like(uu)]));          colors.append([0,200,255])
    faces.append(np.hstack([uu, vv, np.ones_like(uu)]));           colors.append([0,120,240])
    faces.append(np.hstack([uu, np.zeros_like(uu), vv]));          colors.append([0,255,0])
    faces.append(np.hstack([uu, np.ones_like(uu),  vv]));          colors.append([255,0,0])
    faces.append(np.hstack([np.zeros_like(uu), uu, vv]));          colors.append([255,0,255])
    faces.append(np.hstack([np.ones_like(uu),  uu, vv]));          colors.append([0,0,255])
    pts = np.vstack(faces)
    clr = np.vstack([np.tile(np.array(c, dtype=np.float32)[None,:]/255.0, (res*res,1)) for c in colors])
    return pts, clr


def update_cloud():
    global cube_pcd, cube_points_local, cube_colors, R_euler, t, scale
    T = get_transform_mat(R_euler, t, scale)
    pts_h = np.hstack([cube_points_local, np.ones((cube_points_local.shape[0],1))])
    pts_w = (T @ pts_h.T).T[:, :3]
    cube_pcd.points = o3d.utility.Vector3dVector(pts_w)
    cube_pcd.colors = o3d.utility.Vector3dVector(cube_colors)
    vis.update_geometry(cube_pcd)


def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1:
        shift_pressed = True
    elif action == 0:
        shift_pressed = False
    return True


def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cloud()


def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cloud()


def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cloud()


def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cloud()


def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cloud()


def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cloud()


def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cloud()


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

points3D_df = pd.read_pickle("data/points3D.pkl")
scene_pcd = load_point_cloud(points3D_df)
vis.add_geometry(scene_pcd)

axes = load_axes()
vis.add_geometry(axes)

cube_points_local, cube_colors = make_unit_cube_face_points(res=50)
cube_pcd = o3d.geometry.PointCloud()
cube_pcd.points = o3d.utility.Vector3dVector(cube_points_local)
cube_pcd.colors = o3d.utility.Vector3dVector(cube_colors)
vis.add_geometry(cube_pcd)

R_euler = np.array([0,0,0]).astype(float)
t = np.array([0,0,0]).astype(float)
scale = 1.0
update_cloud()

vc = vis.get_view_control()
vc_cam = vc.convert_to_pinhole_camera_parameters()
initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
initial_cam = np.concatenate([initial_cam, np.zeros([1,4])], 0)
initial_cam[-1,-1] = 1.
setattr(vc_cam, 'extrinsic', initial_cam)
vc.convert_from_pinhole_camera_parameters(vc_cam)

shift_pressed = False
vis.register_key_action_callback(340, toggle_key_shift)
vis.register_key_action_callback(344, toggle_key_shift)
vis.register_key_callback(ord('A'), update_tx)
vis.register_key_callback(ord('S'), update_ty)
vis.register_key_callback(ord('D'), update_tz)
vis.register_key_callback(ord('Z'), update_rx)
vis.register_key_callback(ord('X'), update_ry)
vis.register_key_callback(ord('C'), update_rz)
vis.register_key_callback(ord('V'), update_scale)

print('[Keyboard usage]')
print('Translate along X-axis\tA / Shift+A')
print('Translate along Y-axis\tS / Shift+S')
print('Translate along Z-axis\tD / Shift+D')
print('Rotate    along X-axis\tZ / Shift+Z')
print('Rotate    along Y-axis\tX / Shift+X')
print('Rotate    along Z-axis\tC / Shift+C')
print('Scale                 \tV / Shift+V')

vis.run()
vis.destroy_window()

np.save('cube_transform_mat.npy', get_transform_mat(R_euler, t, scale))
np.save('cube_points.npy', np.asarray(cube_pcd.points))
np.save('cube_colors.npy', np.asarray(cube_pcd.colors))

K = np.array([[1868.27, 0.0, 540.0],
              [0.0, 1869.18, 960.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352, 0.0], dtype=np.float64)

IMAGES_DF_PATH = "data/images.pkl"
IMAGES_ROOT    = "data/frames"
VIDEO_OUT      = "/home/ivmlab3/3dcv_hw/homework2-asd30627/ar_output.mp4"
FPS            = 10

T_cube = np.load('cube_transform_mat.npy')
cube_points_local = np.load('cube_points.npy')
cube_colors = np.load('cube_colors.npy')
bgr_per_point = (cube_colors[:, ::-1] * 255.0).astype(np.uint8)

ones = np.ones((cube_points_local.shape[0], 1))
points_world = (T_cube @ np.hstack([cube_points_local, ones]).T).T[:, :3]

if not os.path.exists(IMAGES_DF_PATH):
    raise FileNotFoundError(f"Missing file: {IMAGES_DF_PATH}")
df = pd.read_pickle(IMAGES_DF_PATH)

frames = []
size_wh = None

for _, row in df.iterrows():
    if "IMAGE_PATH" in row and isinstance(row["IMAGE_PATH"], str):
        p = row["IMAGE_PATH"]
    elif "NAME" in row and isinstance(row["NAME"], str):
        p = row["NAME"]
    elif "filename" in row and isinstance(row["filename"], str):
        p = row["filename"]
    elif "path" in row and isinstance(row["path"], str):
        p = row["path"]
    else:
        continue
    img_path = p if os.path.isabs(p) else os.path.join(IMAGES_ROOT, p)

    img = cv.imread(img_path)
    if img is None:
        continue
    if size_wh is None:
        h, w = img.shape[:2]
        size_wh = (w, h)

    if not all(k in row.index for k in ("QX","QY","QZ","QW")):
        frames.append(img)
        continue
    if not all(k in row.index for k in ("TX","TY","TZ")):
        frames.append(img)
        continue

    qx, qy, qz, qw = float(row["QX"]), float(row["QY"]), float(row["QZ"]), float(row["QW"])
    Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
    tcw = np.array([[float(row["TX"])],
                    [float(row["TY"])],
                    [float(row["TZ"])]], dtype=np.float64)

    pts_cam = (Rcw @ points_world.T) + tcw
    Z = pts_cam[2, :]
    mask = Z > 0
    if not np.any(mask):
        frames.append(img)
        continue

    rvec, _ = cv.Rodrigues(Rcw)
    img_pts, _ = cv.projectPoints(points_world[mask], rvec, tcw, K, DIST)
    pts2d = img_pts.reshape(-1, 2).astype(np.int32)
    cols = bgr_per_point[mask]

    canvas = img.copy()
    H, W = canvas.shape[:2]
    for (x, y), c in zip(pts2d, cols):
        if 0 <= x < W and 0 <= y < H:
            cv.circle(canvas, (int(x), int(y)), 2, tuple(int(v) for v in c), -1, lineType=cv.LINE_AA)

    frames.append(canvas)

if frames:
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    vw = cv.VideoWriter(VIDEO_OUT, fourcc, FPS, size_wh)
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[OK] wrote {VIDEO_OUT} with {len(frames)} frames.")
else:
    print("[WARN] no usable frames; check images.pkl and image paths.")