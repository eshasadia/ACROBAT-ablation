import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

import utils as u
import utils_tc as utc


def ecc_registration(source, target, params):
    """
    Intensity-based registration using the Enhanced Correlation Coefficient (ECC)
    algorithm (cv2.findTransformECC).

    Parameters
    ----------
    source : tc.Tensor (1 x C x H x W)
        The source image tensor.
    target : tc.Tensor (1 x C x H x W)
        The target image tensor.
    params : dict
        Registration parameters:
          - echo (bool)
          - registration_size (int): resolution for registration
          - motion_model (str): one of 'translation', 'euclidean', 'affine'
          - num_iterations (int): max ECC iterations (default 1000)
          - termination_eps (float): ECC termination threshold (default 1e-8)

    Returns
    -------
    final_transform : tc.Tensor (1 x 2 x 3)
        The estimated affine transform in theta (normalised) space.
    """
    echo = params['echo']
    resolution = params['registration_size']
    motion_model_name = params.get('motion_model', 'euclidean')
    num_iterations = params.get('num_iterations', 1000)
    termination_eps = params.get('termination_eps', 1e-8)

    motion_model_map = {
        'translation': cv2.MOTION_TRANSLATION,
        'euclidean': cv2.MOTION_EUCLIDEAN,
        'affine': cv2.MOTION_AFFINE,
    }
    motion = motion_model_map.get(motion_model_name, cv2.MOTION_EUCLIDEAN)

    resampled_source, resampled_target = u.initial_resampling(source, target, resolution)
    if echo:
        print(f"Resampled source size: {resampled_source.size()}")
        print(f"Resampled target size: {resampled_target.size()}")

    src = u.tensor_to_image(resampled_source)[:, :, 0]
    trg = u.tensor_to_image(resampled_target)[:, :, 0]
    src = (src * 255).astype(np.uint8)
    trg = (trg * 255).astype(np.uint8)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations, termination_eps)

    try:
        _, warp_matrix = cv2.findTransformECC(trg, src, warp_matrix, motion, criteria)
    except Exception as exc:
        if echo:
            print(f"ECC registration failed ({exc}), using identity.")
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # warp_matrix maps source -> target; invert to get the displacement convention
    final_transform = np.eye(3)
    final_transform[0:2, 0:3] = warp_matrix
    try:
        final_transform = np.linalg.inv(final_transform)
    except Exception:
        final_transform = np.eye(3)
    final_transform = utc.affine2theta(final_transform, (resampled_source.size(2), resampled_source.size(3))).type_as(source).unsqueeze(0)
    if echo:
        print(f"Calculated ECC transform: {final_transform}")
    return final_transform
