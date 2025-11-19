import numpy as np
import open3d as o3d
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import moviepy.editor as mpy
import os
import matplotlib.gridspec as gridspec




def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid

    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    return points, centroid, max_dist


def read_point_cloud(file_path):
    data = np.loadtxt(file_path)
    points = data[:, :3]
    colors = data[:, 3:] / 255.0
    normalized_points, _, _ = normalize_point_cloud(points)
    return normalized_points, colors


def compute_pca(points):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    cov_matrix = np.cov(points_centered.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    return eigenvectors, centroid, points_centered


def align_point_cloud_with_pca(points, eigenvectors, target_axes=None):
    if target_axes is None:
        target_axes = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    rotation_matrix = eigenvectors @ target_axes.T

    u, s, vh = np.linalg.svd(rotation_matrix)
    rotation_matrix = u @ vh

    aligned_points = points @ rotation_matrix

    return aligned_points, rotation_matrix


def rotate_points(points, angle_deg, axis='z'):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'x':
        rot_matrix = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        rot_matrix = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    else:
        rot_matrix = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    return np.dot(points, rot_matrix)


def create_point_cloud_visualization(pcd1, pcd2, rotation_angle, rotation_axis='z'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    plt.subplots_adjust(
        left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.1, hspace=0.02
    )

    def plot_point_cloud(ax, points, colors, rotation_angle):
        rotated_points = rotate_points(points, rotation_angle, axis='z')
        ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                   c=colors, alpha=1, s=1)
        ax.view_init(elev=0, azim=rotation_angle)
        max_range = np.max([np.max(rotated_points[:, i]) - np.min(rotated_points[:, i]) for i in range(3)])
        mid_x = (np.max(rotated_points[:, 0]) + np.min(rotated_points[:, 0])) / 2
        mid_y = (np.max(rotated_points[:, 1]) + np.min(rotated_points[:, 1])) / 2
        mid_z = (np.max(rotated_points[:, 2]) + np.min(rotated_points[:, 2])) / 2
        scale_factor = 0.7
        ax.set_xlim(mid_x - max_range * scale_factor / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range * scale_factor / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range * scale_factor / 2, mid_z + max_range / 2)
        ax.axis('off')

    plot_point_cloud(axes[0], pcd1[0], pcd1[1], rotation_angle)
    plot_point_cloud(axes[1], pcd2[0], pcd2[1], rotation_angle)

    return fig


def generate_animation(pcd1, pcd2, output_file="point_cloud_animation.mp4",
                       frames=180, fps=60):
    print("Generating animation frames...")

    def make_frame(t):
        rotation_angle = (t * fps) % 360
        fig = create_point_cloud_visualization(pcd1, pcd2, rotation_angle)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        img = img.reshape(height, width, 3)
        plt.close(fig)
        return img

    print(f"Generating video: {output_file}")
    animation = mpy.VideoClip(make_frame, duration=frames / fps)
    animation.write_videofile(output_file, fps=fps, codec="libx264", bitrate="10000k")
    print(f"Video generation completed: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate point cloud animation with specified input and output paths.')
    parser.add_argument('--input_paths', nargs=2, required=True,
                        help='Two input point cloud file paths (e.g., --input_paths path1.txt path2.txt)')
    parser.add_argument('--output_path', required=True,
                        help='Output animation file path (e.g., --output_path result.mp4)')

    args = parser.parse_args()
    file_paths = args.input_paths
    output_file = args.output_path

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return

    point_clouds = []
    for file_path in file_paths:
        points, colors = read_point_cloud(file_path)
        point_clouds.append((points, colors))
        print(f"Point cloud read: {file_path}, Number of points: {len(points)}")

    eigenvectors, centroid, _ = compute_pca(point_clouds[0][0])

    target_axes = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    aligned_point_clouds = []
    for points, colors in point_clouds:
        aligned_points, rotation_matrix = align_point_cloud_with_pca(points, eigenvectors, target_axes)
        aligned_point_clouds.append((aligned_points, colors))

    generate_animation(aligned_point_clouds[0], aligned_point_clouds[1], output_file)


if __name__ == "__main__":
    main()