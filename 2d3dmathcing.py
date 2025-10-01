from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import open3d as o3d
from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428)    # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]], dtype=np.float64)
    distCoeffs  = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352], dtype=np.float64)

    # ---- KNN matching + ratio test ----
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        return False, None, None, None

    pts2d = np.float32([kp_query[m.queryIdx] for m in good]).reshape(-1,1,2)
    pts3d = np.float32([kp_model[m.trainIdx] for m in good]).reshape(-1,1,3)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, cameraMatrix, distCoeffs,
        flags=cv2.SOLVEPNP_EPNP, reprojectionError=6.0,
        iterationsCount=300, confidence=0.999
    )
    if (not retval) or (inliers is None) or (len(inliers) < 6):
        return False, None, None, None

    in2d = pts2d[inliers.ravel()]
    in3d = pts3d[inliers.ravel()]
    retval, rvec, tvec = cv2.solvePnP(
        in3d, in2d, cameraMatrix, distCoeffs,
        rvec, tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return retval, rvec, tvec, inliers

def rotation_error(R1, R2):
    #TODO: calculate rotation error
    R_gt  = R.from_quat(R1).as_matrix()[0]
    R_est = R.from_quat(R2).as_matrix()[0]
    R_rel = R_gt.T @ R_est
    cos_theta = (np.trace(R_rel) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def translation_error(t1, t2):
    #TODO: calculate translation error
    return float(np.linalg.norm(t1 - t2))

def visualization(Camera2World_Transform_Matrixs, points3D_df,
                  zoom=0.08,
                  front=(0, 0, -1),
                  up=(0, -1, 0),
                  lookat=None,
                  scale=1.0):
    """
    只做可視化：點雲 + 相機金字塔 + 相機軌跡
    """
    import numpy as np
    import open3d as o3d

    # --- point cloud ---
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.copy())
    pcd.colors = o3d.utility.Vector3dVector(rgb.copy())

    if scale != 1.0:
        center = pcd.get_center()
        pcd.scale(scale, center=center)
    else:
        center = pcd.get_center()

    geoms = [pcd]

    # --- camera frustum (in camera frame) ---
    cam_pts = np.array([
        [0, 0, 0],
        [-1, -0.75, -2],
        [ 1, -0.75, -2],
        [ 1,  0.75, -2],
        [-1,  0.75, -2]
    ], dtype=np.float64) * 0.8
    cam_lines = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]], dtype=np.int32)

    traj = []
    for c2w in Camera2World_Transform_Matrixs:
        homo = np.c_[cam_pts, np.ones((cam_pts.shape[0], 1))]
        cam_world = (c2w @ homo.T).T[:, :3]
        if scale != 1.0:
            cam_world = (cam_world - center) * scale + center

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(cam_world)
        ls.lines  = o3d.utility.Vector2iVector(cam_lines)
        ls.colors = o3d.utility.Vector3dVector(
            np.tile([[1, 0, 0]], (cam_lines.shape[0], 1))
        )
        geoms.append(ls)

        traj.append(cam_world[0])  

    # --- trajectory polyline ---
    if len(traj) >= 2:
        traj = np.asarray(traj)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(traj)
        ls.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(traj)-1)])
        ls.colors = o3d.utility.Vector3dVector(
            np.tile([[0, 1, 0]], (len(traj)-1, 1))
        )
        geoms.append(ls)

    # --- view ---
    if lookat is None:
        lookat = np.mean(np.asarray(pcd.points), axis=0)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Q2-1 Trajectory + PointCloud",
        zoom=zoom,
        front=list(front),
        lookat=list(lookat),
        up=list(up)
    )


def p3p_numeric(Pw, uv, K, max_iters=30, tol=1e-8):

    uv1 = np.hstack([uv, np.ones((3, 1))])                 # (3x3): [u,v,1]
    f = (np.linalg.inv(K) @ uv1.T).T                        # (3x3)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)        

    a = np.linalg.norm(Pw[1] - Pw[2])  # |P2-P3|
    b = np.linalg.norm(Pw[0] - Pw[2])  # |P1-P3|
    c = np.linalg.norm(Pw[0] - Pw[1])  # |P1-P2|
    a2, b2, c2 = a*a, b*b, c*c

    def residuals(lmb):
        P1c = lmb[0]*f[0]; P2c = lmb[1]*f[1]; P3c = lmb[2]*f[2]
        r1 = np.dot(P1c-P2c, P1c-P2c) - c2
        r2 = np.dot(P1c-P3c, P1c-P3c) - b2
        r3 = np.dot(P2c-P3c, P2c-P3c) - a2
        return np.array([r1, r2, r3])

    def jacobian(lmb):
        P1c = lmb[0]*f[0]; P2c = lmb[1]*f[1]; P3c = lmb[2]*f[2]
        x12 = P1c - P2c
        x13 = P1c - P3c
        x23 = P2c - P3c
        J = np.zeros((3, 3))
        J[0,0] =  2*np.dot(x12, f[0]); J[0,1] = -2*np.dot(x12, f[1])
        J[1,0] =  2*np.dot(x13, f[0]); J[1,2] = -2*np.dot(x13, f[2])
        J[2,1] =  2*np.dot(x23, f[1]); J[2,2] = -2*np.dot(x23, f[2])
        return J
  
    base = (a + b + c) / 3.0
    init_scales = [1.0, 0.5, 1.5, 2.0]   
    solutions, seen = [], set()

    for s in init_scales:
        lmb = np.array([base*s, base*s, base*s], dtype=float)
        mu = 1e-2   
        for _ in range(max_iters):
            r = residuals(lmb)
            J = jacobian(lmb)
            H = J.T @ J
            g = J.T @ r
            try:
                delta = np.linalg.solve(H + mu*np.eye(3), -g)
            except np.linalg.LinAlgError:
                break
            lmb_new = lmb + delta
            
            if np.linalg.norm(residuals(lmb_new)) < np.linalg.norm(r):
                lmb = lmb_new
                mu *= 0.5
                if np.linalg.norm(delta) < tol:
                    break
            else:
                mu *= 2.0

        if np.any(lmb <= 0):
            continue

        Pc = (lmb[:, None] * f)                    # (3,3)

        Pw_cent = Pw.mean(axis=0); Pc_cent = Pc.mean(axis=0)
        X = Pw - Pw_cent; Y = Pc - Pc_cent
        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        Rc = Vt.T @ U.T
        if np.linalg.det(Rc) < 0:
            Vt[-1, :] *= -1
            Rc = Vt.T @ U.T
        tc = (Pc_cent - Rc @ Pw_cent).reshape(3, 1)

        zc = (Rc @ Pw.T + tc).T[:, 2]
        if np.all(zc > 0):
            key = tuple(np.round(np.hstack([Rc.flatten(), tc.flatten()]), 6))
            if key not in seen:
                seen.add(key)
                solutions.append((Rc, tc))

    return solutions


def ransac_p3p(query, model, thresh_px=4.0, max_iter=2000, confidence=0.999):

    kp_query, desc_query = query
    kp_model, desc_model = model

    K = np.array([[1868.27,    0.0, 540.0],
                  [   0.0, 1869.18, 960.0],
                  [   0.0,    0.0,   1.0]], dtype=np.float64)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 6:
        return False, None, None, np.array([], dtype=int)

    uv_all = np.float64([kp_query[m.queryIdx] for m in good])  # (N,2)
    Pw_all = np.float64([kp_model[m.trainIdx] for m in good])  # (N,3)
    N = len(Pw_all)

    def project(Rc, tc, Pw):
        Rc = np.asarray(Rc, dtype=float)
        tc = np.asarray(tc, dtype=float).ravel()
        Pw = np.asarray(Pw, dtype=float)

        Pc = Pw @ Rc.T + tc  # (N,3)

        z = Pc[:, 2]
        z_safe = np.clip(z, 1e-12, None)
        x = Pc[:, 0] / z_safe
        y = Pc[:, 1] / z_safe

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        s = K[0, 1]  

        u = fx * x + s * y + cx
        v = fy * y + cy
        uv = np.column_stack((u, v))
        return uv, z

    best_inliers = None
    best_model = None
    max_trials = max_iter
    trials = 0
    rng = np.random.default_rng(1428)

    while trials < max_trials:
        idx = rng.choice(N, size=3, replace=False)
        Pw3 = Pw_all[idx]
        uv3 = uv_all[idx]

        models = p3p_numeric(Pw3, uv3, K)
        if not models:
            trials += 1
            continue

        for (Rc, tc) in models:
            uv_hat, z = project(Rc, tc, Pw_all)
            reproj = np.linalg.norm(uv_hat - uv_all, axis=1)
            inliers = np.where((reproj < thresh_px) & (z > 0))[0]

            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (Rc.copy(), tc.copy())

                w = max(1e-9, len(inliers) / N)
                eps = 1 - w**3
                eps = min(max(eps, 1e-12), 1 - 1e-12)
                max_trials = min(max_iter, int(np.log(1 - confidence) / np.log(eps)) + 1)

        trials += 1

    if best_model is not None and best_inliers is not None and len(best_inliers) >= 3:
        Rc, tc = best_model
        rvec = cv2.Rodrigues(Rc)[0]
        tvec = tc.copy()
        return True, rvec, tvec, best_inliers
    else:
        return False, None, None, np.array([], dtype=int)

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    IMAGE_ID_LIST = [200,201]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        # retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        retval, rvec, tvec, inliers = ransac_p3p((kp_query, desc_query), (kp_model, desc_model), thresh_px=4.0, max_iter=2000, confidence=0.999)

        # print("\nretval\n", retval, "\nrvec\n", rvec, "\ntvec\n", tvec, "\ninliers\n", inliers)
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error
        if retval:
            rotq_est = R.from_rotvec(rvec.reshape(1,3)).as_quat().reshape(1,4)  # [qx,qy,qz,qw]
            tvec_est = tvec.reshape(1,3)
            r_error = rotation_error(rotq_gt, rotq_est)
            t_error = translation_error(tvec_gt, tvec_est)
        else:
            r_error, t_error = np.nan, np.nan

        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # TODO: calculate median of relative rotation angle differences and translation differences and print them
    rot_med = np.nanmedian(np.asarray(rotation_error_list))
    trans_med = np.nanmedian(np.asarray(translation_error_list))
    print(f"Median rotation error (deg): {rot_med:.6f}")
    print(f"Median translation error (m): {trans_med:.6f}")

    # TODO: result visualization
    Camera2World_Transform_Matrixs = []
    for rvec, tvec in zip(r_list, t_list):
        # TODO: calculate camera pose in world coordinate system
        if (rvec is None) or (tvec is None):
            continue
        Rcw, _ = cv2.Rodrigues(rvec)       # world -> camera
        tcw = tvec.reshape(3,1)
        Rwc = Rcw.T
        twc = -Rwc @ tcw
        camera_to_world_matrix = np.eye(4)
        camera_to_world_matrix[:3,:3] = Rwc
        camera_to_world_matrix[:3, 3] = twc.ravel()
        Camera2World_Transform_Matrixs.append(camera_to_world_matrix)

    visualization(Camera2World_Transform_Matrixs, points3D_df,
                  zoom=0.06,        
                  scale=1.0,        
                  front=(0,0,-1),
                  up=(0,-1,0))
