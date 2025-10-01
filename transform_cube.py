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
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def update_cube():
    global cube, cube_vertices, R_euler, t, scale
    transform_mat = get_transform_mat(R_euler, t, scale)
    transform_vertices = (transform_mat @ np.concatenate([
                            cube_vertices.transpose(), 
                            np.ones([1, cube_vertices.shape[0]])
                            ], axis=0)).transpose()
    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)


def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1: # key down
        shift_pressed = True
    elif action == 0: # key up
        shift_pressed = False
    return True


def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()


def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()


def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()


def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()


def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()


# === 互動視窗 ===
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# load point cloud
points3D_df = pd.read_pickle("data/points3D.pkl")
pcd = load_point_cloud(points3D_df)
vis.add_geometry(pcd)

# load axes
axes = load_axes()
vis.add_geometry(axes)

# load cube
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube_vertices = np.asarray(cube.vertices).copy()
vis.add_geometry(cube)

R_euler = np.array([0, 0, 0]).astype(float)
t = np.array([0, 0, 0]).astype(float)
scale = 1.0
update_cube()

# just set a proper initial camera view
vc = vis.get_view_control()
vc_cam = vc.convert_to_pinhole_camera_parameters()
initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
initial_cam[-1, -1] = 1.
setattr(vc_cam, 'extrinsic', initial_cam)
vc.convert_from_pinhole_camera_parameters(vc_cam)

# set key callback
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
np.save('cube_vertices.npy', np.asarray(cube.vertices))

K = np.array([[1868.27, 0.0,   540.0],
              [0.0,    1869.18, 960.0],
              [0.0,       0.0,    1.0]], dtype=np.float64)
DIST = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352, 0.0], dtype=np.float64)

IMAGES_DF_PATH = "data/images.pkl"
IMAGES_ROOT    = "data/frames"
VIDEO_OUT      = "ar_output.mp4"
FPS            = 10

T_cube = np.load('cube_transform_mat.npy')

unit_box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
triangles = np.asarray(unit_box.triangles)    
verts_local = np.asarray(unit_box.vertices)  

ones = np.ones((verts_local.shape[0], 1))
verts_world = (T_cube @ np.hstack([verts_local, ones]).T).T  

face_colors = [
    (0, 200, 255),  
    (0, 120, 240),  
    (0, 255, 0),    
    (255, 0, 0),    
    (255, 0, 255),  
    (0, 0, 255),    
]

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
        print("[WARN] image path column missing; skip this row.")
        continue
    img_path = p if os.path.isabs(p) else os.path.join(IMAGES_ROOT, p)

    img = cv.imread(img_path)
    if img is None:
        print(f"[WARN] cannot open image: {img_path}")
        continue
    if size_wh is None:
        h, w = img.shape[:2]
        size_wh = (w, h)

    if not all(k in row.index for k in ("QX","QY","QZ","QW")):
        print("[WARN] quaternion columns missing; skip this row.")
        frames.append(img)
        continue
    if not all(k in row.index for k in ("TX","TY","TZ")):
        print("[WARN] translation columns missing; skip this row.")
        frames.append(img)
        continue

    qx, qy, qz, qw = float(row["QX"]), float(row["QY"]), float(row["QZ"]), float(row["QW"])
    Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
    tcw = np.array([[float(row["TX"])],
                    [float(row["TY"])],
                    [float(row["TZ"])]], dtype=np.float64)

    verts_cam = (Rcw @ verts_world.T) + tcw   
    Z_all = verts_cam[2, :]

    if not np.any(Z_all > 0):
        frames.append(img)
        continue

    rvec, _ = cv.Rodrigues(Rcw)
    img_pts, _ = cv.projectPoints(verts_world, rvec, tcw, K, DIST)
    pts2d = img_pts.reshape(-1, 2)  # (8, 2)

    tri_depths = []
    for i, tri in enumerate(triangles):
        z_mean = float(np.mean(Z_all[tri]))
        tri_depths.append((z_mean, i))

    tri_depths.sort(key=lambda x: x[0], reverse=True)

    canvas = img.copy()
    color_idx = 0
    for zmean, tri_i in tri_depths:
        tri = triangles[tri_i].astype(int)
        if np.any(Z_all[tri] <= 0):
            continue
        poly = pts2d[tri].astype(np.int32)
        col_fill = face_colors[color_idx % len(face_colors)]
        col_edge = (max(col_fill[0]-0, 0), max(col_fill[1]-80, 0), max(col_fill[2]-15, 0))
        cv.fillConvexPoly(canvas, poly, col_fill)
        cv.polylines(canvas, [poly], True, col_edge, 2, lineType=cv.LINE_AA)
        color_idx += 1

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