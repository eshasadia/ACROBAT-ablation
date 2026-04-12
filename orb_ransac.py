import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

import utils as u
import utils_tc as utc

_RATIO_TEST_THRESHOLD = 0.75
_MIN_MATCHES = 4


def orb_ransac(source, target, params):
    """
    Affine registration using ORB keypoints and RANSAC.

    Parameters
    ----------
    source : tc.Tensor (1 x C x H x W)
        The source image tensor.
    target : tc.Tensor (1 x C x H x W)
        The target image tensor.
    params : dict
        Registration parameters (echo, registration_size, show, num_features).

    Returns
    -------
    final_transform : tc.Tensor (1 x 2 x 3)
        The estimated affine transform in theta (normalised) space.
    """
    echo = params['echo']
    resolution = params['registration_size']
    show = params.get('show', False)
    num_features = params.get('num_features', 4096)

    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    src = u.tensor_to_image(resampled_source)[:, :, 0]
    trg = u.tensor_to_image(resampled_target)[:, :, 0]
    src = (src * 255).astype(np.uint8)
    trg = (trg * 255).astype(np.uint8)

    source_keypoints, source_descriptors, target_keypoints, target_descriptors = detect_and_compute_orb(src, trg, num_features)
    if echo:
        print(f"Number of source keypoints: {len(source_keypoints)}")
        print(f"Number of target keypoints: {len(target_keypoints)}")

    try:
        source_points, target_points = matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors)
    except Exception:
        final_transform = np.eye(3)
        final_transform = utc.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
        return final_transform

    if echo:
        print(f"Number of matched points: {len(source_points)}")

    if show:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(src, cmap='gray')
        plt.plot(source_points[:, 0, 0], source_points[:, 0, 1], "r*")
        plt.subplot(1, 2, 2)
        plt.imshow(trg, cmap='gray')
        plt.plot(target_points[:, 0, 0], target_points[:, 0, 1], "r*")
        plt.show()

    try:
        transform, _ = cv2.estimateAffinePartial2D(source_points, target_points, cv2.RANSAC)
    except Exception:
        transform = np.eye(3)[0:2, :]

    final_transform = np.eye(3)
    final_transform[0:2, 0:3] = transform
    try:
        final_transform = np.linalg.inv(final_transform)
    except Exception:
        final_transform = np.eye(3)
    final_transform = utc.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
    if echo:
        print(f"Calculated transform: {final_transform}")
    return final_transform


def detect_and_compute_orb(source, target, num_features=4096):
    orb = cv2.ORB_create(num_features)
    source_keypoints, source_descriptors = orb.detectAndCompute(source, None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target, None)
    return source_keypoints, source_descriptors, target_keypoints, target_descriptors


def matcher(source_keypoints, target_keypoints, source_descriptors, target_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(source_descriptors, target_descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < _RATIO_TEST_THRESHOLD * n.distance:
            good_matches.append(m)
    if len(good_matches) < _MIN_MATCHES:
        raise ValueError("Not enough good ORB matches found.")
    source_points = np.float32([source_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_points = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return source_points, target_points
