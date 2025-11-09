#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
import random
import uuid
from pathlib import Path
from random import randint
from time import time

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.camera_utils import generate_interpolated_path
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2_torch, focal2fov
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import get_camera_from_tensor
from utils.sfm_utils import save_time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False


def save_pose(path, quat_pose, train_cams, llffhold=2):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]
    
    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)
    
    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(confidence_path, device='cuda', scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
    
    Args:
        confidence_path (str): Path to the .npy confidence file.
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.
    
    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
    
    return lr_modifiers



def load_matrix_argument(arg, expected_shape, name):
    if arg is None:
        return None
    path = Path(str(arg))
    if path.exists():
        if path.suffix in {".npy", ".npz"}:
            data = np.load(path, allow_pickle=False)
            if isinstance(data, np.lib.npyio.NpzFile):
                if not data.files:
                    raise ValueError(f"{name}: npz file {path} is empty")
                mat = data[data.files[0]]
            else:
                mat = data
        else:
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                mat = np.asarray(loaded, dtype=np.float32)
            except json.JSONDecodeError:
                with open(path, 'r') as f:
                    values = [float(x) for x in f.read().replace(';', ',').split(',') if x.strip()]
                mat = np.asarray(values, dtype=np.float32)
    else:
        values = [float(x) for x in str(arg).replace(';', ',').split(',') if x.strip()]
        mat = np.asarray(values, dtype=np.float32)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.size != int(np.prod(expected_shape)):
        raise ValueError(f"{name}: expected {int(np.prod(expected_shape))} values, got {mat.size}")
    mat = mat.reshape(expected_shape)
    return mat



def build_camera_info_from_inputs(image_path, intrinsics, Tcw, uid):
    image_path = Path(image_path)
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    if intrinsics is None or Tcw is None:
        raise ValueError('intrinsics and Tcw must be provided to build CameraInfo')
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)
    Rcw = np.asarray(Tcw[:3, :3], dtype=np.float32)
    tcw = np.asarray(Tcw[:3, 3], dtype=np.float32)
    Rc2w = np.linalg.inv(Rcw).astype(np.float32)
    cam_info = CameraInfo(
        uid=int(uid),
        R=Rc2w,
        T=tcw,
        FovY=fovy,
        FovX=fovx,
        image=image,
        image_path=str(image_path),
        image_name=image_path.stem,
        width=width,
        height=height,
    )
    return cam_info



def apply_mask_to_gaussian_grads(gaussians, mask):
    if mask is None:
        return
    if isinstance(mask, torch.Tensor):
        mask_t = mask.to(gaussians._xyz.device).float()
    else:
        mask_t = torch.tensor(mask, device=gaussians._xyz.device, dtype=torch.float32)

    def _mask_param(param):
        if param.grad is None:
            return
        expanded = mask_t
        while expanded.ndim < param.grad.ndim:
            expanded = expanded.unsqueeze(-1)
        param.grad.mul_(expanded)

    _mask_param(gaussians._xyz)
    _mask_param(gaussians._features_dc)
    _mask_param(gaussians._features_rest)
    _mask_param(gaussians._opacity)
    _mask_param(gaussians._scaling)
    _mask_param(gaussians._rotation)



def zero_out_old_camera_grads(gaussians, old_camera_count):
    if gaussians.P.grad is None:
        return
    if old_camera_count <= 0:
        return
    gaussians.P.grad[:old_camera_count] = 0



def compute_diff_mask(gaussians, baseline_state, original_count, eps=1e-5):
    current_xyz = gaussians._xyz.detach()
    device = current_xyz.device
    mask = torch.zeros(current_xyz.shape[0], dtype=torch.bool, device=device)
    if original_count < current_xyz.shape[0]:
        mask[original_count:] = True
    if original_count > 0:
        base_xyz = baseline_state['xyz'].to(device)
        xyz_diff = torch.norm(current_xyz[:original_count] - base_xyz, dim=1)
        base_dc = baseline_state['features_dc'].to(device)
        base_rest = baseline_state['features_rest'].to(device)
        base_opacity = baseline_state['opacity'].to(device)
        base_scaling = baseline_state['scaling'].to(device)
        base_rotation = baseline_state['rotation'].to(device)
        dc_diff = torch.norm(gaussians._features_dc.detach()[:original_count] - base_dc, dim=(1, 2))
        rest_diff = torch.norm(gaussians._features_rest.detach()[:original_count] - base_rest, dim=(1, 2))
        opacity_diff = torch.norm(gaussians._opacity.detach()[:original_count] - base_opacity, dim=1)
        scaling_diff = torch.norm(gaussians._scaling.detach()[:original_count] - base_scaling, dim=1)
        rotation_diff = torch.norm(gaussians._rotation.detach()[:original_count] - base_rotation, dim=1)
        combined = (xyz_diff > eps) | (dc_diff > eps) | (rest_diff > eps) | (opacity_diff > eps) | (scaling_diff > eps) | (rotation_diff > eps)
        mask[:original_count] |= combined
    return mask



def evaluate_and_save(scene, gaussians, pipe, background, output_dir, cameras, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    residual_dir = os.path.join(output_dir, 'residuals')
    os.makedirs(residual_dir, exist_ok=True)
    metrics = []
    with torch.no_grad():
        for cam in cameras:
            pose = gaussians.get_RT(cam.uid)
            render_pkg = render(cam, gaussians, pipe, background, camera_pose=pose)
            pred = torch.clamp(render_pkg['render'], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to(device), 0.0, 1.0)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0)).item()
            else:
                ssim_value = ssim(pred, gt).item()
            l1_val = l1_loss(pred, gt).item()
            psnr_val = psnr(pred, gt).item()
            diff = torch.abs(pred - gt).mean(dim=0, keepdim=True).repeat(3, 1, 1)
            torchvision.utils.save_image(pred.detach().cpu(), os.path.join(output_dir, f"{cam.image_name}_render.png"))
            torchvision.utils.save_image(gt.detach().cpu(), os.path.join(output_dir, f"{cam.image_name}_gt.png"))
            torchvision.utils.save_image(diff.detach().cpu(), os.path.join(residual_dir, f"{cam.image_name}_residual.png"))
            metrics.append({
                'image_name': cam.image_name,
                'psnr': psnr_val,
                'ssim': ssim_value,
                'l1': l1_val,
            })
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics



def snapshot_gaussians(gaussians):
    return {
        'xyz': gaussians._xyz.detach().clone(),
        'features_dc': gaussians._features_dc.detach().clone(),
        'features_rest': gaussians._features_rest.detach().clone(),
        'opacity': gaussians._opacity.detach().clone(),
        'scaling': gaussians._scaling.detach().clone(),
        'rotation': gaussians._rotation.detach().clone(),
    }



def estimate_pose_with_mast3r(reference_image_paths, new_image_path, device, ckpt_path, image_size=512, schedule='cosine', lr=0.01, niter=300, focal_avg=False):
    from mast3r.model import AsymmetricMASt3R
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.geometry import inv
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from utils.sfm_utils import load_images

    image_files = list(reference_image_paths) + [str(new_image_path)]
    imgs, _ = load_images(image_files, size=image_size, verbose=False)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    model.eval()
    output = inference(pairs, model, device=device, batch_size=1, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr, focal_avg=focal_avg)
    extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
    intrinsics = to_numpy(scene.get_intrinsics())
    return intrinsics[-1], extrinsics_w2c[-1]


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    # per-point-optimizer
    confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy")
    confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))
    scene = Scene(dataset, gaussians)

    if opt.pp_optimizer:
        gaussians.training_setup_pp(opt, confidence_lr)                          
    else:
        gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    train_cams_init = scene.getTrainCameras().copy()
    for save_iter in saving_iterations:
        os.makedirs(scene.model_path + f'/pose/ours_{save_iter}', exist_ok=True)
        save_pose(scene.model_path + f'/pose/ours_{save_iter}/pose_org.npy', gaussians.P, train_cams_init)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
    first_iter += 1
    start = time()
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if opt.optim_pose==False:
            gaussians.P.requires_grad_(False)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        pose = gaussians.get_RT(viewpoint_cam.uid)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        iter_end.record()
        # for param_group in gaussians.optimizer.param_groups:
        #     for param in param_group['params']:
        #         if param is gaussians.P:
        #             print(viewpoint_cam.uid, param.grad)
        #             break
        # print("Gradient of self.P:", gaussians.P.grad)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            # if iteration < opt.densify_until_iter:
                # # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Log and save
            if iteration == opt.iterations:
                end = time()
                train_time_wo_log = end - start
                save_time(scene.model_path, '[2] train_joint_TrainTime', train_time_wo_log)
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_pose(scene.model_path + f'/pose/ours_{iteration}/pose_optimized.npy', gaussians.P, train_cams_init)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    end = time()
    train_time = end - start
    save_time(scene.model_path, '[2] train_joint', train_time)


def incremental_training(dataset, opt, pipe, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy")
    confidence_lr = None
    if os.path.exists(confidence_path):
        confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))

    scene = Scene(dataset, gaussians, shuffle=False)

    if args.new_image is None:
        raise ValueError('Incremental training requires --new_image')
    if not Path(args.new_image).exists():
        raise FileNotFoundError(f'new image not found: {args.new_image}')

    opt.iterations = args.incremental_iters
    opt.position_lr_init *= 0.5
    opt.position_lr_final *= 0.5
    opt.scaling_lr *= 0.5
    opt.feature_lr *= 0.5
    opt.densify_from_iter = max(args.stage1_iters + 1, 1)
    opt.densify_until_iter = max(args.stage2_iters, opt.densify_from_iter)

    if opt.pp_optimizer and confidence_lr is not None:
        gaussians.training_setup_pp(opt, confidence_lr)
    else:
        gaussians.training_setup(opt)

    if not args.load_ckpt:
        raise ValueError('Incremental training requires --load_ckpt')
    checkpoint_data = torch.load(args.load_ckpt)
    if isinstance(checkpoint_data, tuple):
        model_params = checkpoint_data[0]
    else:
        model_params = checkpoint_data
    gaussians.restore(model_params, opt)

    baseline_state = snapshot_gaussians(gaussians)
    original_point_count = gaussians._xyz.shape[0]

    old_camera_count = len(scene.getTrainCameras())
    max_uid = max((info.uid for info in scene.train_camera_infos), default=old_camera_count - 1)

    intrinsics = load_matrix_argument(args.new_K, (3, 3), '--new_K')
    Tcw = load_matrix_argument(args.new_Tcw, (4, 4), '--new_Tcw')

    if args.est_pose:
        if args.mast3r_ckpt is None:
            raise ValueError('--est_pose requires --mast3r_ckpt')
        ref_infos = scene.train_camera_infos[:args.est_max_refs] if args.est_max_refs > 0 else scene.train_camera_infos
        if not ref_infos:
            raise ValueError('No reference images available for pose estimation')
        reference_paths = [info.image_path for info in ref_infos]
        est_K, est_Tcw = estimate_pose_with_mast3r(reference_paths, args.new_image, args.est_device, args.mast3r_ckpt, image_size=args.est_image_size, focal_avg=args.focal_avg)
        if intrinsics is None:
            intrinsics = est_K
        if Tcw is None:
            Tcw = est_Tcw

    if intrinsics is None or Tcw is None:
        raise ValueError('Must provide intrinsics and pose for new image via --new_K/--new_Tcw or --est_pose')

    new_uid = max_uid + 1
    cam_info = build_camera_info_from_inputs(args.new_image, intrinsics, Tcw, new_uid)
    appended = scene.append_train_camera(cam_info)
    new_camera = appended[1.0]
    train_cameras = scene.getTrainCameras()

    new_pose_tensor = get_tensor_from_camera(new_camera.world_view_transform.transpose(0, 1)).cuda()
    gaussians.append_pose(new_pose_tensor)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    stage1_iters = min(args.stage1_iters, args.incremental_iters)
    stage2_iters = min(args.stage2_iters, args.incremental_iters)
    total_iters = args.incremental_iters
    stage2_start = stage1_iters + 1
    stage_visibility_mask = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, total_iters + 1), desc='Incremental training')

    old_cameras = train_cameras[:old_camera_count]

    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        if iteration <= stage1_iters:
            viewpoint_cam = new_camera
        elif iteration <= stage2_iters:
            if iteration % 2 == 0 or not old_cameras:
                viewpoint_cam = new_camera
            else:
                idx = randint(0, len(old_cameras) - 1)
                viewpoint_cam = old_cameras[idx]
        else:
            idx = randint(0, len(train_cameras) - 1)
            viewpoint_cam = train_cameras[idx]

        pose = gaussians.get_RT(viewpoint_cam.uid)
        bg = torch.rand((3), device='cuda') if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image = render_pkg['render']
        viewspace_point_tensor = render_pkg['viewspace_points']
        visibility_filter = render_pkg['visibility_filter']
        radii = render_pkg['radii']
        viewspace_point_tensor.retain_grad()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        if iteration <= stage1_iters:
            vis_mask = visibility_filter.float().to(stage_visibility_mask.device)
            if stage_visibility_mask.shape[0] != vis_mask.shape[0]:
                stage_visibility_mask = torch.zeros_like(vis_mask)
            stage_visibility_mask = torch.maximum(stage_visibility_mask, vis_mask)
            apply_mask_to_gaussian_grads(gaussians, stage_visibility_mask)

        zero_out_old_camera_grads(gaussians, old_camera_count)

        if stage2_start <= iteration <= stage2_iters:
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            if iteration % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        progress_bar.set_postfix({'Loss': f'{ema_loss_for_log:.7f}'})

    progress_bar.close()

    metrics_dir = os.path.join(scene.model_path, 'incremental_metrics')
    evaluate_and_save(scene, gaussians, pipe, background, metrics_dir, train_cameras)

    torch.save((gaussians.capture(), total_iters), os.path.join(scene.model_path, 'chkpnt_incremental.pth'))
    scene.save('incremental')

    if args.save_diff:
        diff_mask = compute_diff_mask(gaussians, baseline_state, original_point_count, eps=args.diff_eps)
        gaussians.save_ply_subset(args.save_diff, diff_mask)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % 5000 == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, camera_pose=pose)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--new_image', type=str, default=None)
    parser.add_argument('--new_K', type=str, default=None)
    parser.add_argument('--new_Tcw', type=str, default=None)
    parser.add_argument('--est_pose', action='store_true')
    parser.add_argument('--mast3r_ckpt', type=str, default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    parser.add_argument('--est_device', type=str, default='cuda')
    parser.add_argument('--est_image_size', type=int, default=512)
    parser.add_argument('--est_max_refs', type=int, default=4)
    parser.add_argument('--incremental_iters', type=int, default=1500)
    parser.add_argument('--stage1_iters', type=int, default=400)
    parser.add_argument('--stage2_iters', type=int, default=800)
    parser.add_argument('--save_diff', type=str, default=None)
    parser.add_argument('--diff_eps', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--focal_avg', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    incremental_mode = args.new_image is not None or args.est_pose
    if not incremental_mode:
        args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if incremental_mode:
        incremental_training(lp.extract(args), op.extract(args), pp.extract(args), args)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
