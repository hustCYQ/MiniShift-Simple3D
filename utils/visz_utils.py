import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import open3d as o3d
import utils.mvtec3d_util as mvt_util
##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick

def plot_sample(imgs, scores_:dict, gts, save_folder, class_name):
    total_number = len(imgs)

    scores = scores_.copy()

    # normalize anomalies
    for k,v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value * 255
        scores[k] = scores[k].astype(np.uint8)

    # draw gts
    mask_imgs = []
    for idx in range(total_number):
        gts_ = gts[idx]
        mask_imgs_ = imgs[idx].copy()
        mask_imgs_[gts_ > 0.5] = (255, 0, 0)
        mask_imgs.append(mask_imgs_)

    # save imgs
    for idx in range(total_number):
        cv2.imwrite(os.path.join(save_folder,f'{class_name}_{idx:03d}.png'),cv2.cvtColor(imgs[idx], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_folder,f'{class_name}_{idx:03d}_gt.png'),cv2.cvtColor(mask_imgs[idx], cv2.COLOR_RGB2BGR))
        for k in scores.keys():
            s = scores[k][idx]
            heat_map = cv2.applyColorMap(s, cv2.COLORMAP_JET)
            visz_map = cv2.addWeighted(heat_map,0.5, imgs[idx], 0.5, 0)
            cv2.imwrite(os.path.join(save_folder, f'{class_name}_{idx:03d}_am_{k}.png'), visz_map)

def organized_pc_to_o3d(organized_pc, rgb_):
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc)
    rgb = rgb_.copy()
    rgb = rgb.reshape(rgb.shape[0] * rgb.shape[1], rgb.shape[2])

    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    o3d_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    nonzero_rgb = rgb[nonzero_indices, :]
    return o3d_pc, nonzero_indices, nonzero_rgb

def plot_single_pcd(organized_pc, rgb, color_map_scores, gt, alpha=0., use_rgb=True)->([np.ndarray], [str]):
    '''

    Args:
        organized_pc:
        color_map_scores:
        gt:

    Returns:

    '''

    o3d_pc, nonzero_indices, nonzero_rgb = organized_pc_to_o3d(organized_pc, rgb_=rgb)

    non_zero_number = len(nonzero_indices)

    if use_rgb:
        origin_colors = nonzero_rgb / 255.
    else:
        origin_colors = [[0.5, 0.5, 0.5]] * non_zero_number
        origin_colors = np.array(origin_colors)

    heatmap_colors = dict()

    for k in color_map_scores.keys():
        color_map_scores[k] = color_map_scores[k].reshape(color_map_scores[k].shape[0] * color_map_scores[k].shape[1], color_map_scores[k].shape[2])
        color_map_scores[k] = color_map_scores[k][nonzero_indices, :]

        heatmap_colors[k] = origin_colors * alpha + color_map_scores[k] * (1 - alpha)

    gt_colors = origin_colors.copy()
    gt_pcd_vector = gt.reshape(gt.shape[0] * gt.shape[1])[nonzero_indices]

    gt_colors[gt_pcd_vector > 0.5, :] = (1., 0, 0) # RGB

    H, W = gt.shape
    H = H * 3
    W = W * 3
    voxel_size = 1.5
    radius_normal = voxel_size * 2

    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)

    colors = []
    keys = []
    for k, v in heatmap_colors.items():
        colors.append(v)
        keys.append(f's_{k}')

    keys.append('ori')
    keys.append('gt')
    colors.append(origin_colors)
    colors.append(gt_colors)

    render_images = []
    for key, color in zip(keys, colors):

        o3d_pc.colors = o3d.utility.Vector3dVector(color)
        vis.add_geometry(o3d_pc)
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.clear_geometries()

        # convert to rgb
        image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)
        render_images.append(image)

    return render_images, keys


def plot_sample_o3d(pcds, rgbs, scores_dict:dict, gts, save_folder, class_name=None, use_rgb=None):
    total_number = len(pcds)

    color_map_scores = [dict() for i in range(total_number)]

    for k, scores in scores_dict.items():
        max_value = np.max(scores)
        min_value = np.min(scores)

        scores = (scores - min_value) / max_value * 255
        scores= scores.astype(np.uint8)

        # apply colormap
        for idx in range(total_number):
            s = cv2.applyColorMap(scores[idx], cv2.COLORMAP_JET)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            s = s / 255.
            color_map_scores[idx][k] = s

    for idx, (organized_pcd, rgb, color_map_score, gt) in enumerate(zip(pcds, rgbs, color_map_scores, gts)):
        render_images, keys = plot_single_pcd(organized_pcd, rgb, color_map_score, gt, use_rgb=use_rgb)
        for image, k in zip(render_images, keys):
            cv2.imwrite(save_folder[:-4]+'_{}.png'.format(k), image)

def plot_anomaly_score_distributions(scores:dict, ground_truths_list, save_folder, class_name):

    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 100000

    for k, v in scores.items():

        layer_score = np.stack(v, axis=0)
        normal_score = layer_score[ground_truths == 0]
        abnormal_score = layer_score[ground_truths != 0]

        plt.clf()
        plt.figure(figsize=(2, 1.5))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        with plt.style.context(['science', 'ieee', 'no-latex']):
            sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                         stat='probability', alpha=.75)
            sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                         stat='probability', alpha = .75)


        # plt.xlim([0, 2])

        # sns.kdeplot(abnormal_score, shade='fill', label='${d(p_a)}$')
        # sns.kdeplot(normal_score, shade='fill', label='${d(p_n)}$')

        save_path = os.path.join(save_folder, f'0_distributions_{class_name}_{k}.jpg')

        plt.savefig(save_path, bbox_inches='tight', dpi=300)

# def plot_anomaly_score_distributions(scores, gts, save_folder, class_name):
#
#     gts = np.stack(gts, axis=0)
#
#     for k, v in scores.items():
#
#         layer_score = np.stack(v, axis=0)
#         normal_score = layer_score[gts == 0]
#         abnormal_score = layer_score[gts != 0]
#
#         plt.clf()
#
#         sns.kdeplot(abnormal_score, shade='fill', label='${d(p_a)}$', color='red')
#         sns.kdeplot(normal_score, shade='fill', label='${d(p_n)}$', color='green')
#
#         save_path = os.path.join(save_folder, f'0_distributions_{class_name}_{k}.jpg')
#
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')


valid_feature_visualization_methods = ['TSNE', 'PCA']

def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):

    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')



def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label==l, 0],
                      feat_proj[label==l, 1],
                      feat_proj[label==l, 2],s=5)

    return ax1

def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label==l, 0],
                      feat_proj[label==l, 1],s=5)

    return ax1