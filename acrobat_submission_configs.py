import paths as p
import cost_functions as cf

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nonrigid_params():
    """Default nonrigid (deformable) registration parameters."""
    nonrigid_params = dict()
    nonrigid_params['device'] = "cuda:0"
    nonrigid_params['echo'] = True
    nonrigid_params['cost_function'] = cf.get_function("ncc_local_tc")
    nonrigid_params['cost_function_params'] = {'win_size': 7}
    nonrigid_params['regularization_function'] = "diffusion_relative_tc"
    nonrigid_params['regularization_function_params'] = dict()
    nonrigid_params['registration_size'] = 2048
    nonrigid_params['num_levels'] = 7
    nonrigid_params['used_levels'] = 7
    nonrigid_params['iterations'] = 7 * [400]
    nonrigid_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]
    nonrigid_params['alphas'] = [1.2, 1.2, 1.2, 1.2, 1.2, 1.0, 0.6]
    return nonrigid_params


def _preprocessing_params():
    """Default preprocessing parameters (same across all ablation configs)."""
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = False
    preprocessing_params['normalization'] = True
    preprocessing_params['pad_to_same_size'] = True
    preprocessing_params['late_resample'] = False
    preprocessing_params['late_resample_ratio'] = 1.0
    preprocessing_params['pad_value'] = 1.0
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True
    return preprocessing_params

def affine_config():
    config = dict()

    ### Affine Params ###
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['registration_sizes'] = [100, 150, 200, 250, 300, 400, 500, 600]
    affine_params['transform_type'] = 'rigid'
    affine_params['keypoint_threshold'] = 0.005
    affine_params['match_threshold'] = 0.3
    affine_params['sinkhorn_iterations'] = 50
    affine_params['show'] = False
    affine_params['angle_step'] = 60
    affine_params['num_features'] = 256
    affine_params['sparse_size'] = 45
    affine_params['keypoint_size'] = 8
    affine_params['device'] = "cuda:0"

    ### Preprocessing ###
    preprocessing_params = dict()
    preprocessing_params['preprocessing_function'] = "basic_preprocessing"
    preprocessing_params['initial_resampling'] = False
    preprocessing_params['normalization'] = True
    preprocessing_params['pad_to_same_size'] = True
    preprocessing_params['late_resample'] = False
    preprocessing_params['late_resample_ratio'] = 1.0
    preprocessing_params['pad_value'] = 1.0
    preprocessing_params['convert_to_gray'] = True
    preprocessing_params['clahe'] = True

    ### General ###
    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "Affine_Validation"
    config['level'] = 4
    config['registration_method'] = "affine"
    config['registration_params'] = dict()
    config['preprocessing_params'] = preprocessing_params

    config['registration_params']['affine_params'] = affine_params
    return config

def affine_nonrigid_config():
    config = dict()

    ### Affine Params ###
    affine_params = affine_config()['registration_params']['affine_params']

    ### Nonrigid Params ###
    nonrigid_params = dict()
    nonrigid_params['device'] = "cuda:0"
    nonrigid_params['echo'] = True
    nonrigid_params['cost_function'] = cf.get_function("ncc_local_tc")
    nonrigid_params['cost_function_params'] = {'win_size' : 7}
    nonrigid_params['regularization_function'] = "diffusion_relative_tc"
    nonrigid_params['regularization_function_params'] = dict()
    nonrigid_params['registration_size'] = 2048
    nonrigid_params['num_levels'] = 7
    nonrigid_params['used_levels'] = 7
    nonrigid_params['iterations'] = 7*[400]
    nonrigid_params['learning_rates'] = [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015]
    nonrigid_params['alphas'] = [1.2, 1.2, 1.2, 1.2, 1.2, 1.0, 0.6]

    ### Preprocessing ###
    preprocessing_params = affine_config()['preprocessing_params']

    ### General ###
    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "Affine_Nonrigid_Validation_Minimal_Example"
    config['level'] = 3
    config['registration_method'] = "affine_iterative_nonrigid"
    config['registration_params'] = dict()
    config['preprocessing_params'] = preprocessing_params

    ### Iterative Affine Params ###
    config['registration_params']['iterative_affine_params'] = dict()
    config['registration_params']['iterative_affine_params']['device'] = "cuda:0"
    config['registration_params']['iterative_affine_params']['echo'] = True
    config['registration_params']['iterative_affine_params']['cost_function'] = cf.get_function("ncc_local_tc")
    config['registration_params']['iterative_affine_params']['cost_function_params'] = {'win_size' : 7}
    config['registration_params']['iterative_affine_params']['registration_size'] = 256
    config['registration_params']['iterative_affine_params']['num_levels'] = 4
    config['registration_params']['iterative_affine_params']['used_levels'] = 4
    config['registration_params']['iterative_affine_params']['iterations'] = [200, 200, 200, 200]
    config['registration_params']['iterative_affine_params']['learning_rate'] = 0.02

    config['registration_params']['affine_params'] = affine_params
    config['registration_params']['nonrigid_params'] = nonrigid_params
    return config


# ---------------------------------------------------------------------------
# Ablation: affine-only baselines
# ---------------------------------------------------------------------------

def affine_iterative_config():
    """
    Rotation-based landmark combination with iterative NCC affine refinement,
    but *without* the subsequent nonrigid step.
    """
    config = dict()
    affine_params = affine_config()['registration_params']['affine_params']

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "Affine_Iterative_Validation"
    config['level'] = 4
    config['registration_method'] = "affine_iterative"
    config['registration_params'] = dict()
    config['preprocessing_params'] = _preprocessing_params()

    config['registration_params']['iterative_affine_params'] = dict()
    config['registration_params']['iterative_affine_params']['device'] = "cuda:0"
    config['registration_params']['iterative_affine_params']['echo'] = True
    config['registration_params']['iterative_affine_params']['cost_function'] = cf.get_function("ncc_local_tc")
    config['registration_params']['iterative_affine_params']['cost_function_params'] = {'win_size': 7}
    config['registration_params']['iterative_affine_params']['registration_size'] = 256
    config['registration_params']['iterative_affine_params']['num_levels'] = 4
    config['registration_params']['iterative_affine_params']['used_levels'] = 4
    config['registration_params']['iterative_affine_params']['iterations'] = [200, 200, 200, 200]
    config['registration_params']['iterative_affine_params']['learning_rate'] = 0.02
    config['registration_params']['affine_params'] = affine_params
    return config



    """Standalone SIFT + RANSAC (single scale, no rotation loop)."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "SIFT_RANSAC_Validation"
    config['level'] = 4
    config['registration_method'] = "sift_ransac_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def orb_ransac_config():
    """Standalone ORB + RANSAC."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['num_features'] = 4096
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "ORB_RANSAC_Validation"
    config['level'] = 4
    config['registration_method'] = "orb_ransac_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def akaze_ransac_config():
    """Standalone AKAZE + RANSAC."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "AKAZE_RANSAC_Validation"
    config['level'] = 4
    config['registration_method'] = "akaze_ransac_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def brisk_ransac_config():
    """Standalone BRISK + RANSAC."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "BRISK_RANSAC_Validation"
    config['level'] = 4
    config['registration_method'] = "brisk_ransac_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def ecc_affine_config():
    """Standalone ECC affine registration."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['motion_model'] = 'affine'
    affine_params['num_iterations'] = 1000
    affine_params['termination_eps'] = 1e-8
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "ECC_Affine_Validation"
    config['level'] = 4
    config['registration_method'] = "ecc_affine_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def ecc_euclidean_config():
    """Standalone ECC euclidean (rigid) registration."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['motion_model'] = 'euclidean'
    affine_params['num_iterations'] = 1000
    affine_params['termination_eps'] = 1e-8
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "ECC_Euclidean_Validation"
    config['level'] = 4
    config['registration_method'] = "ecc_affine_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def superpoint_superglue_config():
    """Standalone SuperPoint + SuperGlue affine registration."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['transform_type'] = 'rigid'
    affine_params['keypoint_threshold'] = 0.005
    affine_params['match_threshold'] = 0.3
    affine_params['sinkhorn_iterations'] = 50
    affine_params['num_features'] = 256
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "SuperPoint_SuperGlue_Validation"
    config['level'] = 4
    config['registration_method'] = "superpoint_superglue_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


def superpoint_ransac_config():
    """Standalone SuperPoint + RANSAC affine registration."""
    config = dict()
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"

    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / "SuperPoint_RANSAC_Validation"
    config['level'] = 4
    config['registration_method'] = "superpoint_ransac_registration"
    config['registration_params'] = {'affine_params': affine_params}
    config['preprocessing_params'] = _preprocessing_params()
    return config


# ---------------------------------------------------------------------------
# Ablation: affine + nonrigid baselines
# ---------------------------------------------------------------------------

def _nonrigid_config_base(method_name, output_dir, affine_params):
    """Shared template for affine + nonrigid ablation configs."""
    config = dict()
    config['input_datapath'] = p.ACROBAT_validation_data_path
    config['input_csv_path'] = p.ACROBAT_validation_data_path / "acrobat_validation_points_public_1_of_1.csv"
    config['output_path'] = p.ACROBAT_results_path / output_dir
    config['level'] = 3
    config['registration_method'] = method_name
    config['registration_params'] = {
        'affine_params': affine_params,
        'nonrigid_params': _nonrigid_params(),
    }
    config['preprocessing_params'] = _preprocessing_params()
    return config


def sift_ransac_nonrigid_config():
    """SIFT + RANSAC initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("sift_ransac_nonrigid", "SIFT_RANSAC_Nonrigid_Validation", affine_params)


def orb_ransac_nonrigid_config():
    """ORB + RANSAC initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['num_features'] = 4096
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("orb_ransac_nonrigid", "ORB_RANSAC_Nonrigid_Validation", affine_params)


def akaze_ransac_nonrigid_config():
    """AKAZE + RANSAC initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("akaze_ransac_nonrigid", "AKAZE_RANSAC_Nonrigid_Validation", affine_params)


def brisk_ransac_nonrigid_config():
    """BRISK + RANSAC initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("brisk_ransac_nonrigid", "BRISK_RANSAC_Nonrigid_Validation", affine_params)


def ecc_nonrigid_config():
    """ECC (affine) initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['motion_model'] = 'affine'
    affine_params['num_iterations'] = 1000
    affine_params['termination_eps'] = 1e-8
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("ecc_nonrigid", "ECC_Nonrigid_Validation", affine_params)


def superpoint_superglue_nonrigid_config():
    """SuperPoint + SuperGlue initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['transform_type'] = 'rigid'
    affine_params['keypoint_threshold'] = 0.005
    affine_params['match_threshold'] = 0.3
    affine_params['sinkhorn_iterations'] = 50
    affine_params['num_features'] = 256
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("superpoint_superglue_nonrigid", "SuperPoint_SuperGlue_Nonrigid_Validation", affine_params)


def superpoint_ransac_nonrigid_config():
    """SuperPoint + RANSAC initialisation followed by nonrigid registration."""
    affine_params = dict()
    affine_params['echo'] = True
    affine_params['registration_size'] = 620
    affine_params['show'] = False
    affine_params['device'] = "cuda:0"
    return _nonrigid_config_base("superpoint_ransac_nonrigid", "SuperPoint_RANSAC_Nonrigid_Validation", affine_params)