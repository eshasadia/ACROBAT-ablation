import sys
current_file = sys.modules[__name__]

import utils as u

import rotated_landmark_based_combination as rlbc
import sift_ransac as sr
import orb_ransac as orb
import akaze_ransac as akaze
import brisk_ransac as brisk
import ecc_registration as ecc
import superpoint_superglue as sg
import superpoint_ransac as spr

### Note ###
# It is assumed that all algorithms in the initial_registration.py return 2x3 transformation matrix in the PyTorch format
############

### Algorithms ###

def identity_initial_registration(source, target, params):
    return u.create_identity_displacement_field(source)

def rotated_landmark_based_combination(source, target, params):
    return rlbc.rotated_landmark_based_combination(source, target, params)

def sift_ransac(source, target, params):
    return sr.sift_ransac(source, target, params)

def orb_ransac(source, target, params):
    return orb.orb_ransac(source, target, params)

def akaze_ransac(source, target, params):
    return akaze.akaze_ransac(source, target, params)

def brisk_ransac(source, target, params):
    return brisk.brisk_ransac(source, target, params)

def ecc_affine(source, target, params):
    return ecc.ecc_registration(source, target, params)

def superpoint_superglue(source, target, params):
    return sg.superpoint_superglue(source, target, params)

def superpoint_ransac(source, target, params):
    return spr.superpoint_ransac(source, target, params)

### Utility ###

def get_function(function_name):
    return getattr(current_file, function_name)