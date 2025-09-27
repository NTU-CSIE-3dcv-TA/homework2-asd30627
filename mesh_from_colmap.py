import os
import argparse
import struct
import numpy as np
import open3d as o3d


def read_points3D_bin(bin_path):
    with open(bin_path, "rb") as f:
        data = f.read()

    ofs = 0
    def read(fmt):
        nonlocal ofs
        size = struct.calcsize(fmt)
        out = struct.unpack_from(fmt, data, ofs)
        ofs += size
        return out if len(out) > 1 else out[0]

    num_points = read("<Q")
    xyz_list = []

    for _ in range(num_points):
        _pid = read("<Q")
        x, y, z = read("<3d")
        _r, _g, _b = read("<3B")
        _err = read("<d")
        track_len = read("<Q")
        ofs += track_len * struct.calcsize("<II")
        xyz_list.append([x, y, z])

    xyz = np.array(xyz_list, dtype=np.float64)
    return xyz


def make_o3d_pcd_from_xyz(xyz, voxel_size=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 3 if voxel_size else 0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def poisson_reconstruct(pcd, depth=10, density_trim=0.05, target_tris=None, smooth_iter=3):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    dens = np.asarray(densities)
    if 0.0 < density_trim < 0.5:
        keep = dens > np.quantile(dens, density_trim)
        mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if smooth_iter and smooth_iter > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iter)
    if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    mesh.compute_vertex_normals()
    return mesh


def bpa_reconstruct(pcd, radius, target_tris=None, smooth_iter=1):
    radii = o3d.utility.DoubleVector([radius, radius * 2.0, radius * 4.0])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if smooth_iter and smooth_iter > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iter)
    if target_tris and target_tris > 0 and len(mesh.triangles) > target_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    mesh.compute_vertex_normals()
    return mesh


def visualize_geometries(*geoms, window_name="Open3D Viewer",
                         width=1920, height=1080, point_size=3.0,
                         bg=(1.0, 1.0, 1.0), save_png=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    for g in geoms:
        vis.add_geometry(g)
    ropt = vis.get_render_option()
    ropt.background_color = np.asarray(bg, dtype=np.float64)
    ropt.point_size = float(point_size)
    ropt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    vis.poll_events(); vis.update_renderer()
    if save_png:
        vis.capture_screen_image(save_png, do_render=True)
        print(f"[OK] Saved screenshot: {save_png}")
    vis.run()
    vis.destroy_window()



def main():
    ap = argparse.ArgumentParser(description="COLMAP -> Open3D mesh reconstruction")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--colmap_model_dir", type=str,
                     help="Directory containing points3D.bin (e.g., . or ./sparse)")
    src.add_argument("--input_ply", type=str, help="Path to existing point cloud .ply file")

    ap.add_argument("--voxel", type=float, default=0.01,
                    help="Voxel downsampling size in meters before reconstruction. 0 means no downsampling.")
    ap.add_argument("--method", type=str, choices=["poisson", "bpa"], default="poisson")

    ap.add_argument("--poisson_depth", type=int, default=10,
                    help="Poisson octree depth (commonly 8–12)")
    ap.add_argument("--density_trim", type=float, default=0.05,
                    help="Density quantile threshold for trimming thin surfaces (0–0.5)")

    ap.add_argument("--bpa_radius", type=float, default=0.02,
                    help="BPA sphere radius (adjust based on scene scale)")

    ap.add_argument("--target_tris", type=int, default=300000,
                    help="Target number of triangles after simplification (0 means no simplification)")
    ap.add_argument("--out", type=str, default="mesh_out.ply",
                    help="Output mesh file path (.ply/.obj/.stl)")
    ap.add_argument("--no_view", action="store_true", help="Disable visualization preview")

    args = ap.parse_args()

    if args.colmap_model_dir:
        pts_path = os.path.join(args.colmap_model_dir, "points3D.bin")
        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"Not found: {pts_path}")
        xyz = read_points3D_bin(pts_path)
        pcd = make_o3d_pcd_from_xyz(xyz, voxel_size=args.voxel if args.voxel > 0 else None)
    else:
        if not os.path.exists(args.input_ply):
            raise FileNotFoundError(args.input_ply)
        pcd = o3d.io.read_point_cloud(args.input_ply)
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel * 3 if args.voxel else 0.05, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(30)

    if args.method == "poisson":
        mesh = poisson_reconstruct(
            pcd,
            depth=args.poisson_depth,
            density_trim=args.density_trim,
            target_tris=args.target_tris
        )
    else:
        mesh = bpa_reconstruct(
            pcd,
            radius=args.bpa_radius,
            target_tris=args.target_tris
        )

    o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"[OK] Mesh saved to: {args.out}, triangles={len(mesh.triangles)}")

    if not args.no_view:
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        pcd.paint_uniform_color([0.2, 0.6, 1.0])
        visualize_geometries(pcd, mesh, window_name="Q1-2 Mesh Preview")


if __name__ == "__main__":
    main()
