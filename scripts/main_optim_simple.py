import os
from os import walk
from pathlib import Path
import logging
import sys
import time
from dgl.geometry import farthest_point_sampler
from submodules.smplx import smplx
import torch.nn as nn
from pytorch3d.loss import (
    chamfer_distance,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures import Pointclouds
import argparse
from scripts.simple import SMPLXSimp
from utils.functions import *
from utils.util_texture import inpaint_interpolation, apply_lama
from tqdm import tqdm

# from deepface import DeepFace
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

_ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_ROOT_DIR))

# Configure logging
logging.basicConfig(
    filename='../application.log',  # Name of the log file
    level=logging.INFO,  # Minimum severity level to capture (INFO and above)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    filemode='w'
)

body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                         20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70,
                         21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75] + list(
    np.arange(76, 127)), dtype=np.int32)

body_mapping1 = np.array(list(np.arange(76, 127)), dtype=np.int32)
id_hand_twist = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 39, 42]
id_hand_coller = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 40, 43]


def recalage_rigide(model, optimizer0, scheduler0, betas, expression, body_pose, global_orient, left_hand_pose,
                    right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, path):
    patience = 10  # Number of epochs to wait for loss improvement
    min_delta = 0.0001  # Minimum change in loss to be considered as an improvement
    best_loss = float('inf')
    epochs_without_improvement = 0

    total_iterations = 100

    # Initialize tqdm
    with tqdm(total=total_iterations, desc="recalage_rigide", unit="iter") as pbar:
        for i in range(total_iterations):
            optimizer0.zero_grad()
            output = model(betas=betas, expression=expression, body_pose=body_pose, transl=t_params,
                           global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                           return_verts=True)
            if i == 0:
                mesh_temp = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),
                                            faces=model.faces)
                # mesh_temp.export(path + '/smplx_after_box.obj')
            # loss=torch.norm(betas[0,1:])/5+MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,15,16,21,24]],pose_prior[:,[0,15,16,21,24]]) + MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,22]].mean(1))
            loss = chamfer_distance(smplxsimp.apply_indices(output.vertices.to(device)), pointcloud)[0] + MSE_loss(
                output.joints.to(device)[:, body_mapping][:,
                [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 24]],
                pose_prior[:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 24]]) + 10 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18]],
                pose_prior[:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18]]) + MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [22, 23, 24]].mean(1),
                pose_prior[:, [22, 23, 24]].mean(1)) + MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [19, 20, 21]].mean(1), pose_prior[:, [19, 20, 21]].mean(1))
            # print(i, loss)
            # Update tqdm bar
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
            pbar.update(1)
            loss.backward()
            optimizer0.step()

            if best_loss - loss > min_delta:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler0.step(loss)

            # Check if training should stop
            if epochs_without_improvement == patience:
                print("Early stopping. No improvement in loss.")
                break

    # mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export(path + '/smplx_after_rigid.obj')


def pose_optimization(step_pose_L_sc, model, optimizer, scheduler, betas, expression, body_pose, global_orient,
                      left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, path):
    total_iterations = 250
    # Initialize tqdm
    with tqdm(total=total_iterations, desc="pose optim", unit="iter") as pbar:
        for i in range(total_iterations):
            optimizer.zero_grad()
            output = model(betas=betas, expression=expression, body_pose=body_pose, transl=t_params,
                           global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                           return_verts=True)

            if step_pose_L_sc:
                loss_sc = 10 * torch.norm(body_pose.reshape((21, 3))[[5, 8, 9, 10, 11, 12, 13, 14], :])
            else:
                loss_sc = 0.0

            loss = loss_sc + MSE_loss(output.joints.to(device)[:, body_mapping][:,
                                      [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]], pose_prior[:,
                                                                                                     [0, 2, 3, 4, 5, 6,
                                                                                                      7, 9, 10, 11, 12,
                                                                                                      13, 14, 15, 16,
                                                                                                      17,
                                                                                                      18]]) + 2 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [22, 23, 24]].mean(1),
                pose_prior[:, [22, 23, 24]].mean(1)) + 2 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [19, 20, 21]].mean(1), pose_prior[:, [19, 20, 21]].mean(1))
            # print(i, loss)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

    # mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export(path + '/smplx_after_pose.obj')


def shape_optimization(step_shape_L_chamfer, step_shape_L_P2S, step_shape_L_sc, model, optimizer1, optimizer2,
                       scheduler1, scheduler2, betas, expression, body_pose, global_orient, left_hand_pose,
                       right_hand_pose, t_params, pose_prior, pointcloud, PC, PC1, body_mapping, id_hand_twist,
                       id_hand_coller, body_mapping1, path):
    patience = 10  # Number of epochs to wait for loss improvement
    min_delta = 0.001  # Minimum change in loss to be considered as an improvement
    best_loss = float('inf')
    epochs_without_improvement = 0
    total_iterations = 250
    with tqdm(total=total_iterations, desc="shape optim step 1", unit="iter") as pbar:
        for i in range(total_iterations):
            optimizer1.zero_grad()
            output = model(betas=betas, expression=expression, body_pose=body_pose, transl=t_params,
                           global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                           return_verts=True)

            smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
                               faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
                                   device)])

            if step_shape_L_chamfer:
                loss_chamfer = chamfer_distance(smplxsimp.apply_indices(output.vertices.to(device)), pointcloud)[0]
            else:
                loss_chamfer = 0.0
            loss = loss_chamfer + MSE_loss(output.joints.to(device)[:, body_mapping][:,
                                           [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 24]],
                                           pose_prior[:,
                                           [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21,
                                            24]]) + 10 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18]],
                pose_prior[:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18]]) + MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [22, 23, 24]].mean(1),
                pose_prior[:, [22, 23, 24]].mean(1)) + MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [19, 20, 21]].mean(1), pose_prior[:, [19, 20, 21]].mean(1))
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)
            loss.backward()

            optimizer1.step()
            if best_loss - loss > min_delta:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler1.step(loss)
            # Check if training should stop
            if epochs_without_improvement == patience:
                print("Early stopping. No improvement in loss.")
                break
    # mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export(path + '/smplx_after_shape_s1.obj')

    patience = 10  # Number of epochs to wait for loss improvement
    min_delta = 1  # Minimum change in loss to be considered as an improvement
    best_loss = float('inf')
    epochs_without_improvement = 0

    total_iterations = 600
    with tqdm(total=total_iterations, desc="shape optim step 2", unit="iter") as pbar:
        for i in range(total_iterations):
            optimizer2.zero_grad()
            output = model(betas=betas, expression=expression, body_pose=body_pose, transl=t_params,
                           global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                           return_verts=True)

            smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
                               faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
                                   device)])
            if step_shape_L_P2S:
                loss_P2S = knn_loss1(smpl_mesh, PC)
            else:
                loss_P2S = 0.0

            if step_shape_L_sc:
                hand_penal = loss_with_constraint(left_hand_pose, -0.8, 0.5, 1e6) + loss_with_constraint(
                    right_hand_pose, -0.8, 0.5, 1e6) + torch.norm(left_hand_pose[0, id_hand_twist]) * 1e3 + torch.norm(
                    left_hand_pose[0, id_hand_coller]) * 1e2 + torch.norm(
                    right_hand_pose[0, id_hand_twist]) * 1e3 + torch.norm(right_hand_pose[0, id_hand_coller]) * 1e2
            else:
                hand_penal = 0.0

            if step_shape_L_chamfer:
                loss_chamfer = 10 * chamfer_distance(smplxsimp.apply_indices(output.vertices.to(device)), pointcloud)[0]
            else:
                loss_chamfer = 0.0

            loss_face = 1e4 * MSE_loss(output.joints.to(device)[:, body_mapping1], pose_prior[:, 44:])
            loss = hand_penal + loss_face + loss_chamfer + loss_P2S + 1e4 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:,
                [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, ]],
                pose_prior[:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]) + 1e3 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [22, 23, 24]].mean(1),
                pose_prior[:, [22, 23, 24]].mean(1)) + 1e3 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [19, 20, 21]].mean(1), pose_prior[:, [19, 20, 21]].mean(1))
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)
            loss.backward()
            optimizer2.step()

            if best_loss - loss > min_delta:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler2.step(loss)

            # Check if training should stop
            if epochs_without_improvement == patience:
                print("Early stopping. No improvement in loss.")
                break
    # mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export(path + '/smplx_after_shape_s2.obj')

    total_iterations = 800
    with tqdm(total=total_iterations, desc="shape optim step 3", unit="iter") as pbar:
        for i in range(total_iterations):
            optimizer2.zero_grad()

            output = model(betas=betas, expression=expression, body_pose=body_pose, transl=t_params,
                           global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                           return_verts=True)

            smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
                               faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
                                   device)])
            if step_shape_L_P2S:
                loss_P2S = knn_loss1(smpl_mesh, PC1)
            else:
                loss_P2S = 0.0

            loss_face = MSE_loss(output.joints.to(device)[:, body_mapping1], pose_prior[:, 44:])

            if step_shape_L_sc:
                hand_penal = loss_with_constraint(left_hand_pose, -0.8, 0.5, 1e7) + torch.norm(
                    left_hand_pose[0, id_hand_twist]) * 1e2 + torch.norm(
                    left_hand_pose[0, id_hand_coller]) * 1e1 + torch.norm(
                    right_hand_pose[0, id_hand_twist]) * 1e2 + torch.norm(right_hand_pose[0, id_hand_coller]) * 1e1
            else:
                hand_penal = 0.0

            loss = hand_penal + 1e4 * loss_face + loss_P2S + 1e4 * MSE_loss(output.joints.to(device)[:, body_mapping][:,
                                                                            [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14,
                                                                             15, 16, 17, 18, ]], pose_prior[:,
                                                                                                 [0, 2, 3, 4, 5, 6, 7,
                                                                                                  9, 10, 11, 12, 13, 14,
                                                                                                  15, 16, 17,
                                                                                                  18]]) + 1e3 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [22, 23, 24]].mean(1),
                pose_prior[:, [22, 23, 24]].mean(1)) + 1e3 * MSE_loss(
                output.joints.to(device)[:, body_mapping][:, [19, 20, 21]].mean(1), pose_prior[:, [19, 20, 21]].mean(1))

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)
            loss.backward()
            optimizer2.step()
    # mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export(path + '/smplx_after_shape_s3.obj')

# def deformation_clothes(step_D_L_P2S, step_D_L_laplacien, step_D_L_normal, step_D_L_id, step_D_L_id_face, model, optimizer4, optimizer5, scheduler4, scheduler5, PC1, PC_full, D, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, path):

#   total_iterations = 200
#   with tqdm(total=total_iterations, desc="deformation vector step 1", unit="iter") as pbar:
#     for i in range(total_iterations):
#       optimizer4.zero_grad()

#       output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
#                 return_verts=True, D=D)
#     #   smpl_mesh=Meshes(verts=[(output.vertices.squeeze()).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
#       smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
#                             faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
#                                 device)])
#       smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])

#       loss_P2S = knn_loss2(smpl_mesh,PC1)
#       if step_D_L_laplacien:
#         loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
#       else:
#         loss_laplacian = 0.0
      
#       if step_D_L_normal:
#         loss_normal=mesh_normal_consistency(smpl_mesh_reg)
#       else:
#         loss_normal = 0.0
      
#       if step_D_L_id:
#         loss_id = torch.norm(torch.norm(D, dim=1))
#       else:
#         loss_id = 0.0
#       if step_D_L_id_face:
#         loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
#       else:
#         loss_id_face = 0.0
#       if step_D_L_P2S:
#         loss_P2S = knn_loss2(smpl_mesh,PC1)
#       else:
#         loss_P2S = 0.0

#       loss=2*loss_P2S  +    1e4*loss_laplacian+   1e4*loss_normal + loss_id + loss_id_face

#       pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
#       pbar.update(1)
#       loss.backward()
#       optimizer4.step()
#       scheduler4.step(loss_P2S)

#   mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy(),faces=model.faces)
#   # mesh_f.export(path+ '/smplx_after_deformation_step_1.obj')

#   patience = 10  # Number of epochs to wait for loss improvement
#   min_delta = 0.01  # Minimum change in loss to be considered as an improvement
#   best_loss = float('inf')
#   epochs_without_improvement = 0

#   total_iterations = 1000
#   with tqdm(total=total_iterations, desc="deformation vector step 2", unit="iter") as pbar:
#     for i in range(total_iterations):

#       optimizer5.zero_grad()

#       output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
#                 return_verts=True, D=D)
#     #   smpl_mesh=Meshes(verts=[(output.vertices.squeeze()).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
#       smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
#                     faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
#                         device)])
#       smpl_mesh_reg=Meshes(verts=[output.vertices[0,idx_regularisation].squeeze()], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])
      
#       if step_D_L_P2S:
#         loss_P2S = knn_loss2(smpl_mesh,PC1)
#       else:
#         loss_P2S = 0.0
      
#       if step_D_L_laplacien:
#         loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
#       else:
#         loss_laplacian = 0.0
      
#       if step_D_L_normal:
#         loss_normal=mesh_normal_consistency(smpl_mesh_reg)
#       else:
#         loss_normal = 0.0
      
#       if step_D_L_id:
#         loss_id = torch.norm(torch.norm(D, dim=1))
#       else:
#         loss_id = 0.0
#       if step_D_L_id_face:
#         loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
#       else:
#         loss_id_face = 0.0
#       loss= 2*loss_P2S + 1e3 * loss_laplacian +   1e3*loss_normal + loss_id + loss_id_face


#       pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
#       pbar.update(1)

#       loss.backward()
#       optimizer5.step()
#       if best_loss - loss_P2S > min_delta:
#             best_loss = loss_P2S
#             epochs_without_improvement = 0
#       else:
#           epochs_without_improvement += 1

#       scheduler5.step(loss_P2S)

#       # Check if training should stop
#       if epochs_without_improvement == patience:
#           print("Early stopping. No improvement in loss.")
#           break
#   mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy(),faces=model.faces)
#   # mesh_f.export(path+ '/smplx_after_deformation_step_2.obj')


#   total_iterations = 3000
#   patience = 10  # Number of epochs to wait for loss improvement
#   min_delta = 0.01  # Minimum change in loss to be considered as an improvement
#   best_loss = float('inf')
#   epochs_without_improvement = 0
#   optimizer5 = torch.optim.Adam([D], lr=1e-5)

#   with tqdm(total=total_iterations, desc="deformation vector step 3", unit="iter") as pbar:
#     for i in range(total_iterations):
#       optimizer5.zero_grad()

#       output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
#                 return_verts=True, D=D)

#       smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
#                     faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
#                         device)])
#       smpl_mesh_reg=Meshes(verts=[output.vertices[0,idx_regularisation].squeeze()], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])
      
#       if step_D_L_P2S:
#         loss_P2S = knn_loss3(smpl_mesh,PC_full)  +  15 * knn_loss2(smpl_mesh,PC1)
#       else:
#         loss_P2S = 0.0
      
#       if step_D_L_laplacien:
#         loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
#       else:
#         loss_laplacian = 0.0
      
#       if step_D_L_normal: 
#         loss_normal=mesh_normal_consistency(smpl_mesh_reg)
#       else:
#         loss_normal = 0.0
      
#       if step_D_L_id:
#         loss_id =  torch.norm(torch.norm(D, dim=1))
#       else:
#         loss_id = 0.0

#       if step_D_L_id_face:
#         loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
#       else:
#         loss_id_face = 0.0

    
#       if i < 1000:
#         loss=2.5*loss_P2S  +  2 * 1e4 * loss_laplacian+   1e5 * loss_normal +  loss_id + loss_id_face
#       elif i < 1500:
#         loss=2.5*loss_P2S  + 1e4 * loss_laplacian+   5 * 1e4 * loss_normal +  loss_id + loss_id_face
#       elif i < 2000:
#         loss=2.5*loss_P2S  +  5 * 1e3 * loss_laplacian+  1e4 *loss_normal +  loss_id + loss_id_face
#       elif i < 2500:
#         loss=2.5*loss_P2S  +  1e3 * loss_laplacian+   1e4 * loss_normal +  loss_id + loss_id_face
#       else:
#         loss=2.5*loss_P2S  + 5 *  1e2 * loss_laplacian+  5 * 1e3 * loss_normal +  loss_id + loss_id_face

#       pbar.set_postfix({"Loss": f"{loss_P2S.item():.4f}"})
#       pbar.update(1)
      
#       loss.backward()
#       optimizer5.step()
#       if best_loss - loss_P2S > min_delta:
#             best_loss = loss_P2S
#             epochs_without_improvement = 0
#       else:
#           epochs_without_improvement += 1

#       scheduler5.step(loss_P2S)

#   mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy(),faces=model.faces)
#   # mesh_f.export(path+ '/smplx_after_deformation_step_3.obj')
#   return mesh_f

def deformation_clothes(step_D_L_P2S, step_D_L_laplacien, step_D_L_normal, step_D_L_id, step_D_L_id_face, model, optimizer4, optimizer5, scheduler4, scheduler5, PC1, PC_full, D, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, path):

  total_iterations = 1000
  with tqdm(total=total_iterations, desc="deformation vector step 1", unit="iter") as pbar:
    for i in range(total_iterations):
      optimizer4.zero_grad()

      output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                return_verts=True, D=D)
      smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
                            faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
                                device)])
    #   smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
      smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])

      # loss_P2S = knn_loss2(smpl_mesh,PC1)
      if step_D_L_laplacien:
        loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
      else:
        loss_laplacian = 0.0
      
      if step_D_L_normal:
        loss_normal=mesh_normal_consistency(smpl_mesh_reg)
      else:
        loss_normal = 0.0
      
      if step_D_L_id:
        loss_id = torch.norm(torch.norm(D, dim=1))
      else:
        loss_id = 0.0
      if step_D_L_id_face:
        loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
      else:
        loss_id_face = 0.0
      if step_D_L_P2S:
        loss_P2S = knn_loss2(smpl_mesh,PC1)
      else:
        loss_P2S = 0.0

      loss=2*loss_P2S  +    1e4*loss_laplacian+   1e4*loss_normal + loss_id + loss_id_face #+ chamfer_distance(smplxsimp.apply_indices(output.vertices.to(device)), PC1)[0]/10

      pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
      pbar.update(1)
      loss.backward()
      optimizer4.step()
      scheduler4.step(loss_P2S)

  mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy(),faces=model.faces)
  # mesh_f.export(path+ '/smplx_after_deformation_step_1.obj')

  patience = 10  # Number of epochs to wait for loss improvement
  min_delta = 0.01  # Minimum change in loss to be considered as an improvement
  best_loss = float('inf')
  epochs_without_improvement = 0

  total_iterations = 1500
  with tqdm(total=total_iterations, desc="deformation vector step 2", unit="iter") as pbar:
    for i in range(total_iterations):

      optimizer5.zero_grad()

      output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                return_verts=True, D=D)
      smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
                            faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
                                device)])
    #   smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
      smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])
      
      if step_D_L_P2S:
        loss_P2S = knn_loss2(smpl_mesh,PC1)
      else:
        loss_P2S = 0.0
      
      if step_D_L_laplacien:
        loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
      else:
        loss_laplacian = 0.0
      
      if step_D_L_normal:
        loss_normal=mesh_normal_consistency(smpl_mesh_reg)
      else:
        loss_normal = 0.0
      
      if step_D_L_id:
        loss_id = torch.norm(torch.norm(D, dim=1))
      else:
        loss_id = 0.0
      if step_D_L_id_face:
        loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
      else:
        loss_id_face = 0.0
      loss= 2*loss_P2S + 1e1*loss_laplacian+   1e2*loss_normal + loss_id + loss_id_face# +  chamfer_distance(smplxsimp.apply_indices(output.vertices.to(device)), PC1)[0]/10


      pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
      pbar.update(1)

      loss.backward()
      optimizer5.step()
      if best_loss - loss_P2S > min_delta:
            best_loss = loss_P2S
            epochs_without_improvement = 0
      else:
          epochs_without_improvement += 1

      scheduler5.step(loss_P2S)

      # Check if training should stop
      if epochs_without_improvement == patience:
          print("Early stopping. No improvement in loss.")
          break

  total_iterations = 500
  with tqdm(total=total_iterations, desc="deformation vector step 3", unit="iter") as pbar:
    for i in range(total_iterations):
      optimizer5.zero_grad()

      output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                return_verts=True, D=D)
      smpl_mesh = Meshes(verts=[smplxsimp.apply_indices(output.vertices.to(device)).squeeze()],
                            faces=[(torch.tensor(smplxsimp.get_faces().astype(np.float64), dtype=torch.int32)).to(
                                device)])
    #   smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
      smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])
      
      if step_D_L_P2S:
        loss_P2S = knn_loss2(smpl_mesh,PC1)
      else:
        loss_P2S = 0.0
      
      if step_D_L_laplacien:
        loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
      else:
        loss_laplacian = 0.0
      
      if step_D_L_normal: 
        loss_normal=mesh_normal_consistency(smpl_mesh_reg)
      else:
        loss_normal = 0.0
      
      if step_D_L_id:
        loss_id =  torch.norm(torch.norm(D, dim=1))
      else:
        loss_id = 0.0

      if step_D_L_id_face:
        loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
      else:
        loss_id_face = 0.0
      if step_D_L_P2S:
        loss_P2S = knn_loss2(smpl_mesh,PC1)
      else:
        loss_P2S = 0.0
    

      loss=2.5*loss_P2S  +    1e1*loss_laplacian+   1e2*loss_normal +  loss_id + loss_id_face #+  chamfer_distance(smplxsimp.apply_indices(output.vertices.to(device)), PC1)[0]/10

      pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
      pbar.update(1)
      
      loss.backward()
      optimizer5.step()
      if best_loss - loss_P2S > min_delta:
            best_loss = loss_P2S
            epochs_without_improvement = 0
      else:
          epochs_without_improvement += 1

      scheduler5.step(loss_P2S)

      # Check if training should stop
      if epochs_without_improvement == patience:
          print("Early stopping. No improvement in loss.")
          break
  mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy(),faces=model.faces)
  # mesh_f.export(path+ '/smpl_final_clothes.obj')
  return mesh_f



def main_sample(abs_path, out_path, root_path, model_folder,
                model_type='smplx',
                ext='npz',
                gender='neutral',
                num_betas=70,
                num_expression_coeffs=10,
                use_face_contour=False):
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs, create_left_hand_pose=True,
                         create_right_hand_pose=True, use_pca=False,
                         ext=ext).to(device)
    betas, body_pose, expression, global_orient, s1_params, rx, ry, rz, D, left_hand_pose, right_hand_pose = init_params(
        model, device)

    output = model(betas=betas, expression=expression, body_pose=body_pose,
                   return_verts=True)

    pifu_mesh = trimesh.load_mesh(os.path.join(abs_path, names))
    pifu_mesh.export(out_path + '/pifu.obj')

    verts, faces, aux = load_obj(os.path.join(abs_path, names))

    pose_prior = torch.tensor(np.expand_dims(np.load(os.path.join(abs_path, names[:-4] + '.npy')), axis=0),
                              dtype=torch.float).to(device)
    pointcloud_full = torch.tensor(verts, dtype=torch.float).to(device)
    PC_full = Pointclouds(points=list(pointcloud_full.unsqueeze(0))).to(device)

    idx = farthest_point_sampler(verts.unsqueeze(0), output.vertices.shape[1])
    pointcloud = torch.tensor(np.expand_dims(verts[idx.squeeze()], axis=0), dtype=torch.float).to(device)
    PC = Pointclouds(points=list(pointcloud)).to(device)

    idx = farthest_point_sampler(verts.unsqueeze(0), output.vertices.shape[1] * 2)
    pointcloud1 = torch.tensor(np.expand_dims(verts[idx.squeeze()], axis=0), dtype=torch.float).to(device)
    PC1 = Pointclouds(points=list(pointcloud1)).to(device)

    downsample_mesh1 = trimesh.Trimesh(vertices=None, faces=None)
    downsample_mesh1.vertices = torch.squeeze(pointcloud).cpu().detach().numpy()
    # downsample_mesh1.export(path + '/downsample_pifu.ply')

    downsample_mesh1.vertices = torch.squeeze(pointcloud1).cpu().detach().numpy()
    # downsample_mesh1.export(path + '/downsample_pifu_2.ply')

    # save_obj(path + '/smpl_initial.obj', output.vertices.detach().squeeze(),
    #          (torch.tensor(model.faces.astype(np.int32), dtype=torch.int64)))

    t_params = -recalage_centroide(pointcloud.squeeze(), output.vertices.detach().squeeze()).reshape((1, 3))
    t_params.requires_grad = True
    t_params = t_params.to(device)

    optimizer0 = torch.optim.Adam([t_params, global_orient, betas], lr=1e-3)
    optimizer = torch.optim.Adam([t_params, body_pose, global_orient], lr=0.0001)
    optimizer1 = torch.optim.Adam([t_params, body_pose, global_orient, betas, expression], lr=0.01)
    optimizer2 = torch.optim.Adam(
        [t_params, body_pose, global_orient, betas, expression, left_hand_pose, right_hand_pose], lr=0.005)
    optimizer3 = torch.optim.Adam([body_pose, betas], lr=0.0001)
    optimizer4 = torch.optim.Adam([D], lr=5 * 1e-5)
    optimizer5 = torch.optim.Adam([D], lr=1e-4)

    scheduler0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer0, mode='min', factor=0.5, patience=5,
                                                            verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                           verbose=True)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=5,
                                                            verbose=True)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=5,
                                                            verbose=True)
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='min', factor=0.5, patience=5,
                                                            verbose=True)
    scheduler4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='min', factor=0.5, patience=5,
                                                            verbose=True)
    scheduler5 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer5, mode='min', factor=0.5, patience=5,
                                                            verbose=True)

    # Start the timer
    start_time = time.time()
    logging.info(f"recalage_rigide")
    recalage_rigide(model, optimizer0, scheduler0, betas, expression, body_pose, global_orient, left_hand_pose,
                    right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, out_path)
    end_time = time.time()
    # Calculate the duration
    duration = end_time - start_time
    time_execution[0].append(duration)
    print(f"The process of recalage rigide took {duration} seconds.")
    logging.info(f"The process of recalage rigide took {duration} seconds.")
    # Log the duration to a file
    with open("time_log.txt", "a") as log_file:
        log_file.write(f"recalage_rigide: {duration} seconds\n")
    if step_pose:
        start_time = time.time()
        logging.info(f"pose_optimization")
        pose_optimization(step_pose_L_sc, model, optimizer, scheduler, betas, expression, body_pose, global_orient,
                          left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, out_path)
        end_time = time.time()
        # Calculate the duration
        duration = end_time - start_time
        time_execution[1].append(duration)
        print(f"The process of pose optimization took {duration} seconds.")
        logging.info(f"The process of pose optimization took {duration} seconds.")
        with open("time_log.txt", "a") as log_file:
            log_file.write(f"pose_optimization: {duration} seconds\n")

    start_time = time.time()

    logging.info(f"shape_optimization")
    shape_optimization(step_shape_L_chamfer, step_shape_L_P2S, step_shape_L_sc, model, optimizer1, optimizer2,
                       scheduler1, scheduler2, betas, expression, body_pose, global_orient, left_hand_pose,
                       right_hand_pose, t_params, pose_prior, pointcloud, PC, PC1, body_mapping, id_hand_twist,
                       id_hand_coller, body_mapping1, out_path)
    end_time = time.time()
    # Calculate the duration
    duration = end_time - start_time
    time_execution[2].append(duration)
    print(f"The process of shape optimization took {duration} seconds.")
    logging.info(f"The process of shape optimization took {duration} seconds.")
    with open("time_log.txt", "a") as log_file:
        log_file.write(f"shape_optimization: {duration} seconds\n")


    logging.info(f"deformation_clothes")
    # Start the timer
    start_time = time.time()
    mesh_f = deformation_clothes(step_D_L_P2S, step_D_L_laplacien, step_D_L_normal, step_D_L_id, step_D_L_id_face,
                                 model, optimizer4, optimizer5, scheduler4, scheduler5, PC1, PC_full, D, betas,
                                 expression,
                                 body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior,
                                 pointcloud, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, out_path)
    end_time = time.time()
    # Calculate the duration
    duration = end_time - start_time
    time_execution[3].append(duration)
    print(f"The process of deformation took {duration} seconds.")
    logging.info(f"The process of deformation took {duration} seconds.")
    with open("time_log.txt", "a") as log_file:
        log_file.write(f"deformation: {duration} seconds\n")
    

    start_time = time.time()

    # mesh_simple = trimesh.load_mesh('/data/smplx_uv_simple.obj', process=False, maintain_order=True)
    mesh_f_simple = trimesh.Trimesh(vertices=mesh_f.vertices, faces=mesh_simple.faces, process=False, maintain_order=True)
    mesh_f_simple.export(out_path + '/smpl_final_clothes_simple.obj')
    extract_texture_pifu(pifu_mesh, mesh_f_simple, os.path.join(root_path, args.data_path, 'smplx_uv_simple.obj'), path,
                            uv_size=1024, name="texture_simple.png")
    # extract_texture_pifu(pifu_mesh, mesh_f, os.path.join(root_path, args.data_path, 'smplx_uv.obj'), out_path,
    #                      uv_size=1024)
    # t_params,global_orient,left_hand_pose,right_hand_pose
    data = {'gender': gender, 'scale': s1_params, 'beta': betas, 'theta': body_pose, 'expression': expression,
            'D': D, 't_params': t_params, 'global_orient': global_orient, 'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose}
    np.save(out_path + '/' + names[:-4] + '_data.npy', data)

    end_time = time.time()
    # Calculate the duration
    duration = end_time - start_time
    time_execution[4].append(duration)
    print(f"The process of texture extraction took {duration} seconds.")
    logging.info(f"The process of texture extraction took {duration} seconds.")
    with open("time_log.txt", "a") as log_file:
        log_file.write(f"texture_extraction: {duration} seconds\n")

    # end_time = time.time()
    start_time = time.time()
    path_texture = os.path.join(out_path, "texture_simple.png")
    path_texture_out = os.path.join(out_path, "texture_interpolation_simple.png")
    interpolated_image, mask_finale = inpaint_interpolation(path_texture, path_texture_out)
    end_time = time.time()
    # Calculate the duration
    duration = end_time - start_time
    print(f"The process of texture completion took {duration} seconds.")
    logging.info(f"The process of texture completion took {duration} seconds.")

    if use_lama:
        path_mask = os.path.join(root_path, args.data_path, "combined_mask.png")
        path_lama = os.path.join(out_path, "texture_lama2.png")
        apply_lama(path_texture_out, path_mask, path_lama)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Render SMPLX models with textures")
    parser.add_argument('--output_path', type=str, default='output/recons',
                        help='Output directory for reconstructed mesh')
    parser.add_argument('--data_path', type=str, default='data', help='Directory for target data')
    parser.add_argument('--input_path', type=str, default='output/output-pifu', help='Directory for input data')
    return parser.parse_args(args)


if __name__ == "__main__":

    root_path = _ROOT_DIR.parent
    args = parse_args()

    MSE_loss = nn.MSELoss()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    read_dictionary = np.load(os.path.join(root_path, args.data_path, 'sub_target.npy'), allow_pickle='TRUE').item()
    idx_D = read_dictionary['idx']
    value_to_check = 5813
    idx_D = idx_D[idx_D != value_to_check]

    read_dictionary = np.load(os.path.join(root_path, args.data_path, 'feet_penal.npy'), allow_pickle='TRUE').item()
    idx_feet = read_dictionary['idx']

    idx_feet_complementaire = sorted(set(range(10436)))
    idx_feet_complementaire = torch.tensor(idx_feet_complementaire, dtype=torch.long)

    read_dictionary = np.load(os.path.join(root_path, args.data_path, 'not_penelized.npy'), allow_pickle='TRUE').item()
    idx_not_penelized = read_dictionary['idx']
    read_dictionary = np.load(os.path.join(root_path, args.data_path, 'index_regularisation.npy'),
                              allow_pickle='TRUE').item()
    idx_regularisation = read_dictionary['idx']
    faces_regularisation = read_dictionary['faces']

    smplxsimp = SMPLXSimp(os.path.join(root_path, args.data_path, 'idx_wo_toes.npy'))
    mesh_simple = trimesh.load_mesh(os.path.join(root_path, args.data_path, 'smplx_uv_simple.obj'), process=False, maintain_order=True)

    abs_path = os.path.join(root_path, args.input_path)

    abs_path_target = os.path.join(root_path, args.data_path, 'target')
    abs_path_target_mesh = os.path.join(root_path, args.data_path, 'target_mesh')
    out_path_recons = os.path.join(root_path, args.output_path)

    if not os.path.exists(os.path.join(root_path, args.output_path)):
        os.makedirs(os.path.join(root_path, args.output_path), exist_ok=True)
    if not os.path.exists(out_path_recons):
        os.mkdir(out_path_recons, exist_ok=True)

    model_folder = os.path.join(root_path, 'models')
    model_type = 'smplx'
    plot_joints = True
    use_face_contour = False
    gender = "female"

    ext = 'npz'
    plotting_module = 'pyrender'
    num_betas = 300
    num_expression_coeffs = 300
    sample_shape = True
    sample_expression = True

    # reconstruction //////////////////////////////////////////////////////////////////////////////////////
    step_pose = True
    step_pose_L_sc = True

    # step_shape = True
    step_shape_L_chamfer = True
    step_shape_L_P2S = True
    step_shape_L_sc = True
    # step-D
    step_D_L_laplacien = True
    step_D_L_normal = True
    step_D_L_id = True
    step_D_L_id_face = True
    step_D_L_P2S = True

    use_lama = False


    global time_execution
    time_execution = [[] for i in range(5)]

    for (dirpath, dirnames, filenames) in walk(abs_path):

        for names in filenames:
            if names.endswith(".obj"):
                if names[7:13]=='female':
                    gender='female'
                else:
                    gender='male'
                print(names, gender)
                logging.info(f"Processing file: {names}")
                path = os.path.join(out_path_recons, names[:-4])
                if not os.path.exists(path):
                    os.mkdir(path)
                main_sample(abs_path, path, root_path, model_folder, model_type, ext=ext, gender=gender,
                            num_betas=num_betas,
                            num_expression_coeffs=num_expression_coeffs,
                            use_face_contour=use_face_contour)



    with open("time_log.txt", "a") as log_file:
        log_file.write(f"time_execution: {time_execution}\n")

    # Calculate the mean of each list in time_execution and save it to the log file
    mean_execution_times = [sum(times) / len(times) if times else 0 for times in time_execution]

    with open("time_log.txt", "a") as log_file:
      log_file.write(f"mean_execution_times: {mean_execution_times}\n")
