from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.structures import Pointclouds
import numpy as np
# from sklearn.neighbors import NearestNeighbors
from dgl.geometry import farthest_point_sampler
import os.path as osp
import trimesh
import numpy as np
import torch
import smplx
from os import walk
from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_face_distance,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import torch
import smplx
import torch.nn as nn
from functions import *
from util_texture import inpaint_interpolation, apply_lama
import os
import os.path as osp
from os import walk
import numpy as np
from dgl.geometry import farthest_point_sampler
import os.path as osp
import argparse
import trimesh
import cv2
from pytorch3d.transforms import RotateAxisAngle
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, transforms
import torch
from pytorch3d.io import load_obj,save_obj
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    SoftSilhouetteShader,
    OpenGLOrthographicCameras,
    PointLights,
    Materials,
    TexturesVertex,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV
)


from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
import lpips
import os
# from torchmetrics.functional import ssim, psnr
from PIL import Image
from torchvision import transforms
# transform = torch.nn.functional.to_tensor
# torchvision.transforms.functional.pil_to_tensor
import numpy as np
import cv2
# from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from os import walk
import csv
import torch
import trimesh
from pytorch3d.io import load_obj,save_obj
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from pytorch3d.loss import (
    chamfer_distance,
    mesh_normal_consistency,
)
import os
from sklearn.neighbors import NearestNeighbors
import torch
from trimesh.exchange.load import Trimesh
import numpy as np
import csv
from pytorch3d.structures import Meshes

def normal_consistency_metric_nn(mesh1, mesh2):
    """
    Compute the normal consistency between two meshes using nearest-neighbor matching.

    Parameters:
    - mesh1, mesh2: trimesh.Trimesh objects

    Returns:
    - normal_consistency: float, the normal consistency metric
    """

    # Compute face normals and centroids for both meshes
    normals1 = mesh1.face_normals
    normals2 = mesh2.face_normals
    centroids1 = mesh1.triangles_center
    centroids2 = mesh2.triangles_center

    # Fit nearest neighbors model to the centroids of mesh2
    nn_model = NearestNeighbors(n_neighbors=1).fit(centroids2)

    # Find nearest neighbors in mesh2 for each face in mesh1
    distances, indices = nn_model.kneighbors(centroids1)

    # Extract corresponding normals from mesh2
    corresponding_normals2 = normals2[indices.flatten()]

    # Compute the dot product between corresponding normals
    dot_products = np.einsum('ij,ij->i', normals1, corresponding_normals2)

    # Compute the angular deviation in radians
    angular_deviation = np.arccos(np.clip(dot_products, -1.0, 1.0))

    # Compute the normal consistency metric
    normal_consistency = np.mean(np.abs(np.cos(angular_deviation)))

    return normal_consistency

def calculate_iou(predicted_mask, target_mask):
    # Calculate intersection (logical AND)
    intersection = torch.logical_and(predicted_mask, target_mask).sum().float()

    # Calculate union (logical OR)
    union = torch.logical_or(predicted_mask, target_mask).sum().float()

    return intersection / union

def recalage_rigide(model, optimizer0, scheduler0, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, path):
  print('RECALAGE RIGIDE///////////////////////////////////////////////////////////////////////////////////////////////')

  patience = 10  # Number of epochs to wait for loss improvement
  min_delta = 0.0001  # Minimum change in loss to be considered as an improvement
  best_loss = float('inf')
  epochs_without_improvement = 0

  for i in range(100):
    optimizer0.zero_grad()
    output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose=right_hand_pose,
              return_verts=True)

    # loss=torch.norm(betas[0,1:])/5+MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,15,16,21,24]],pose_prior[:,[0,15,16,21,24]]) + MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,22]].mean(1))
    loss =  chamfer_distance(output.vertices.to(device),pointcloud)[0] +  MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]])  + 10*MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]])+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
    print(i, loss)
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

  mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
  mesh_f.export(path+'/smpl_med0.obj')

def pose_optimization(step_pose_L_sc, model, optimizer, scheduler, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, path):

  print('3D pose optimization////////////////////////////////////////////////////////////////////////////////////////////////////////')

  for i in range(150):
    optimizer.zero_grad()
    output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
              return_verts=True)
    
    if step_pose_L_sc:
      loss_sc = 10*torch.norm(body_pose.reshape((21,3))[[5,8,9,10,11,12,13,14],:]) 
    else:
      loss_sc = 0.0
  
    loss= loss_sc + MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]])+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
    print(i, loss)
    loss.backward()
    optimizer.step()

    scheduler.step(loss)


  mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
  mesh_f.export(path+ '/smpl_med.obj')
  print(path+ "/smpl_med.obj")


def shape_optimization(step_shape_L_chamfer, step_shape_L_P2S, step_shape_L_sc, model, optimizer1, optimizer2, scheduler1, scheduler2, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, PC, PC1, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, path):
  patience = 10  # Number of epochs to wait for loss improvement
  min_delta = 0.001  # Minimum change in loss to be considered as an improvement
  best_loss = float('inf')
  epochs_without_improvement = 0    
  print('shape optimization ///////////////////////////////////////////////////////////////////////////////////////////////')
  for i in range(100):
    optimizer1.zero_grad()
    output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
              return_verts=True)

    smpl_mesh=Meshes(verts=[output.vertices.squeeze().to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
    
    if step_shape_L_chamfer:
       loss_chamfer = chamfer_distance(output.vertices.to(device),pointcloud)[0]
    else:
        loss_chamfer = 0.0
    loss =  loss_chamfer +  MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]])  + 10*MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]])+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
    # loss =  chamfer_distance(output.vertices.to(device),pointcloud)[0] +  5* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]])  + 10*MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]])+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
    # loss = 1e4*loss_face+ chamfer_distance(output.vertices.to(device),y)[0] +  MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]],pose_prior1[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]])  + 10*MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]],pose_prior1[:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]])+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior1[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior1[:,[19,20,21]].mean(1))
    print(i, loss)
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
  mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
  mesh_f.export(path+ '/smpl_med1.obj')



  patience = 10  # Number of epochs to wait for loss improvement
  min_delta = 1  # Minimum change in loss to be considered as an improvement
  best_loss = float('inf')
  epochs_without_improvement = 0

  # print('3D pose and shape optim (Shamfer distance,point2surface loss)///////////////////////////////////////////////////////////////////////////////////////////////')
  for i in range(100):
    optimizer2.zero_grad()
    output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
              return_verts=True)

    smpl_mesh=Meshes(verts=[output.vertices.squeeze().to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
    if step_shape_L_P2S:
      loss_P2S = knn_loss1(smpl_mesh,PC)
    else:
      loss_P2S = 0.0

    if step_shape_L_sc:
      hand_penal = loss_with_constraint( left_hand_pose, -0.8, 0.5, 1e6) + loss_with_constraint( right_hand_pose, -0.8, 0.5, 1e6) +torch.norm(left_hand_pose[0,id_hand_twist])*1e3 +torch.norm(left_hand_pose[0,id_hand_coller])*1e2  + torch.norm(right_hand_pose[0,id_hand_twist])*1e3 + torch.norm(right_hand_pose[0,id_hand_coller])*1e2
    else:
      hand_penal = 0.0
    # hand_penal=loss_with_constraint( left_hand_pose, -0.8, 0.5, 1e6) + loss_with_constraint( right_hand_pose, -0.8, 0.5, 1e6) +torch.norm(left_hand_pose[0,id_hand_twist])*1e3 +torch.norm(left_hand_pose[0,id_hand_coller])*1e2  + torch.norm(right_hand_pose[0,id_hand_twist])*1e3 + torch.norm(right_hand_pose[0,id_hand_coller])*1e2
    # hand_penal = torch.tensor(0.0)

    if step_shape_L_chamfer:
      loss_chamfer = 10*chamfer_distance(output.vertices.to(device),pointcloud)[0]
    else:
       loss_chamfer = 0.0
       
    loss_face=1e4*MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:])
    loss=  hand_penal + loss_face + loss_chamfer + loss_P2S + 1e4* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]) + 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
    # losss1=10*chamfer_distance(output.vertices.to(device),pointcloud)[0]
    # loss_face=1e4*MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:])
    # loss=  hand_penal +loss_face+ losss1 + losss + 1e4* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]) + 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
    print(i, loss)
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
  mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
  mesh_f.export(path+ '/smpl_final.obj')

  # print('3D pose and shape optim point2surface loss)///////////////////////////////////////////////////////////////////////////////////////////////')
  for i in range(100):
    optimizer2.zero_grad()

    output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
              return_verts=True)

    smpl_mesh=Meshes(verts=[output.vertices.squeeze().to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
    if step_shape_L_P2S:
      loss_P2S = knn_loss1(smpl_mesh,PC1)
    else:
      loss_P2S = 0.0

    loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:])

    if step_shape_L_sc:
      hand_penal=loss_with_constraint( left_hand_pose, -0.8, 0.5, 1e7) + torch.norm(left_hand_pose[0,id_hand_twist])*1e2 +torch.norm(left_hand_pose[0,id_hand_coller])*1e1  + torch.norm(right_hand_pose[0,id_hand_twist])*1e2 + torch.norm(right_hand_pose[0,id_hand_coller])*1e1
    else:
      hand_penal = 0.0
    
    
    loss= hand_penal +  1e4*loss_face + loss_P2S + 1e4* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]) + 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))

    print(i, loss)
    loss.backward()
    optimizer2.step()
  mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
  mesh_f.export(path+ '/smpl_final1.obj')

def deformation_clothes(step_D_L_P2S, step_D_L_laplacien, step_D_L_normal, step_D_L_id, step_D_L_id_face, model, optimizer4, optimizer5, scheduler4, scheduler5, PC1, D, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, path):

  print('D clothes deformation///////////////////////////////////////////////////////////////////////////////////////////////')
  output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
              return_verts=True)
  vv1, ff1 = trimesh.remesh.subdivide(output.vertices.squeeze().detach().cpu().numpy(), model.faces)
  mesh_sub = trimesh.Trimesh(vv1, ff1, process=False, maintain_order=True)
  mesh_sub.export(path+ '/smpl_sub.obj')
  vertice_torch = torch.tensor(mesh_sub.vertices, dtype=torch.float32).to(device)
  #faces_torch = torch.tensor(mesh_sub.faces, dtype=torch.int32).to(device)

  for i in range(200):
    optimizer4.zero_grad()

    #output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
    #          return_verts=True)
    smpl_mesh=Meshes(verts=[(vertice_torch+D).to(device)], faces=[(torch.tensor(ff1.astype(np.float64),dtype=torch.int32)).to(device)])
    #smpl_mesh_reg=Meshes(verts=[(vertice_torch[idx_regularisation].squeeze()+D[idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(ff1.astype(np.float64),dtype=torch.int32)).to(device)])

    loss_P2S = knn_loss2(smpl_mesh,PC1)
    if step_D_L_laplacien:
      loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
    else:
      loss_laplacian = 0.0
    
    #if step_D_L_normal:
       #loss_normal=mesh_normal_consistency(smpl_mesh_reg)
    #else:
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

    loss=2*loss_P2S  +    1e4*loss_laplacian+   1e4*loss_normal + loss_id + loss_id_face

    print(i, loss)
    loss.backward()
    optimizer4.step()
    scheduler4.step(loss_P2S)

  #mesh_f = trimesh.Trimesh(vertices=(vertice_torch.squeeze()+D).detach().cpu().numpy(),faces=model.faces)
  mesh_f = trimesh.Trimesh(vertices=(vertice_torch.squeeze()+D).detach().cpu().numpy(), faces=ff1, process=False, maintain_order=True)
  mesh_f.export(path+ '/smpl_final_clothes_0_sub.obj')


  patience = 10  # Number of epochs to wait for loss improvement
  min_delta = 0.01  # Minimum change in loss to be considered as an improvement
  best_loss = float('inf')
  epochs_without_improvement = 0
  # print('D clothes deformation 1///////////////////////////////////////////////////////////////////////////////////////////////')
  for i in range(1000):
    optimizer5.zero_grad()

    #output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
    #          return_verts=True)
    #smpl_mesh=Meshes(verts=[(vertice_torch+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
    #smpl_mesh_reg=Meshes(verts=[(vertice_torch[idx_regularisation].squeeze()+D[idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])
    smpl_mesh=Meshes(verts=[(vertice_torch+D).to(device)], faces=[(torch.tensor(ff1.astype(np.float64),dtype=torch.int32)).to(device)])
    if step_D_L_P2S:
      loss_P2S = knn_loss2(smpl_mesh,PC1)
    else:
      loss_P2S = 0.0
    
    if step_D_L_laplacien:
      loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
    else:
      loss_laplacian = 0.0
    
    #if step_D_L_normal:
    #  loss_normal=mesh_normal_consistency(smpl_mesh_reg)
    #else:
    loss_normal = 0.0
    
    if step_D_L_id:
      loss_id = torch.norm(torch.norm(D, dim=1))
    else:
      loss_id = 0.0
    if step_D_L_id_face:
      loss_id_face = 1e6 * torch.norm(torch.norm(D[idx_D], dim=1))
    else:
      loss_id_face = 0.0
    loss= 2*loss_P2S + 1e1*loss_laplacian+   1e2*loss_normal + loss_id + loss_id_face


    print(i,loss)
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


  # print('D clothes deformation 2///////////////////////////////////////////////////////////////////////////////////////////////')
  for i in range(100):
    optimizer5.zero_grad()

    #output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
    #          return_verts=True)
    #smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
    #smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()+D[idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])
    smpl_mesh=Meshes(verts=[(vertice_torch+D).to(device)], faces=[(torch.tensor(ff1.astype(np.float64),dtype=torch.int32)).to(device)])
    if step_D_L_P2S:
      loss_P2S = knn_loss2(smpl_mesh,PC1)
    else:
      loss_P2S = 0.0
    
    if step_D_L_laplacien:
      loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
    else:
      loss_laplacian = 0.0
    
    #if step_D_L_normal: 
    #  loss_normal=mesh_normal_consistency(smpl_mesh_reg)
    #else:
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
  

    loss=2.5*loss_P2S  +    1e1*loss_laplacian+   1e2*loss_normal +  loss_id + loss_id_face

    print(i,loss)
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
  mesh_f = trimesh.Trimesh(vertices=(vertice_torch.squeeze()+D).detach().cpu().numpy(), faces=ff1, process=False, maintain_order=True)
  mesh_f.export(path+ '/smpl_final_clothes.obj')

  

  return mesh_f




def main(abs_path, out_path, model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=70,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):

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

    i=0

    id_hand_twist=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 39, 42]
    id_hand_coller=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 40, 43]
    for (dirpath, dirnames, filenames) in walk(abs_path):

      for names in filenames:
        if names.endswith(".obj"):
          print(names)
          if names[7:13]=='female':
            gender='female'
          else:
            gender='male'
          i=i+1

          # if i<18:
          #   continue
          # path = os.path.join("/content/drive/MyDrive/Recherche_FARES_MALLEK_final4/X_avatar/prediction/output", names[:-4])
          path = os.path.join(out_path, names[:-4])
          if not os.path.exists(path):
            os.mkdir(path)

          model = smplx.create(model_folder, model_type=model_type,
                              gender=gender, use_face_contour=use_face_contour,
                              num_betas=num_betas,
                              num_expression_coeffs=num_expression_coeffs,create_left_hand_pose=True,create_right_hand_pose=True,use_pca=False,
                              ext=ext).to(device)
          betas,body_pose,expression,global_orient,s1_params,rx,ry,rz,D,left_hand_pose,right_hand_pose= init_params(model, device)
          D=torch.zeros((41853,3), dtype=torch.float32 , device=device, requires_grad=True)
          output = model(betas=betas, expression=expression,body_pose=body_pose,
                        return_verts=True)


          pifu_mesh=trimesh.load_mesh(abs_path+names)
          pifu_mesh.export(path+'/pifu.obj')

          verts, faces, aux = load_obj(abs_path+names)

          pose_prior=torch.tensor(np.expand_dims(np.load( abs_path + names[:-4] + '.npy'), axis=0),dtype=torch.float).to(device)

          idx=farthest_point_sampler(torch.tensor(verts).unsqueeze(0), output.vertices.shape[1])
          pointcloud=torch.tensor(np.expand_dims(verts[idx.squeeze()], axis=0),dtype=torch.float).to(device)
          PC=Pointclouds(points=list(pointcloud)).to(device)

          idx=farthest_point_sampler(torch.tensor(verts).unsqueeze(0), output.vertices.shape[1]*2)
          pointcloud1=torch.tensor(np.expand_dims(verts[idx.squeeze()], axis=0),dtype=torch.float).to(device)
          PC1=Pointclouds(points=list(pointcloud1)).to(device)

          # idx=result=farthest_point_sampler(torch.tensor(verts).unsqueeze(0), output.vertices.shape[1]*3)
          # pointcloud2=torch.tensor(np.expand_dims(verts[idx.squeeze()], axis=0),dtype=torch.float).to(device)
          # PC2=Pointclouds(points=list(pointcloud2)).to(device)

          downsample_mesh1=trimesh.Trimesh(vertices=None,faces=None)
          downsample_mesh1.vertices=torch.squeeze(pointcloud).cpu().detach().numpy()
          downsample_mesh1.export(path+'/downsample_pifu.ply')

          downsample_mesh1.vertices=torch.squeeze(pointcloud1).cpu().detach().numpy()
          downsample_mesh1.export(path+'/downsample_pifu_2.ply')

          save_obj(path+'/smpl_initial.obj',output.vertices.detach().squeeze(), (torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)))

          t_params=-recalage_centroide(pointcloud.squeeze(),output.vertices.detach().squeeze()).reshape((1,3))
          t_params.requires_grad=True
          t_params=t_params.to(device)

          optimizer0 = torch.optim.Adam([t_params,global_orient,betas], lr=1e-3)
          optimizer = torch.optim.Adam([t_params,body_pose,global_orient], lr=0.0001)
          optimizer1 = torch.optim.Adam([t_params,body_pose,global_orient,betas,expression], lr=0.01)
          optimizer2 = torch.optim.Adam([t_params,body_pose,global_orient,betas,expression,left_hand_pose,right_hand_pose], lr=0.005)
          optimizer3 = torch.optim.Adam([body_pose,betas], lr=0.0001)
          optimizer4 = torch.optim.Adam([D], lr=1e-7)
          optimizer5 = torch.optim.Adam([D], lr=1e-4)


          # Define the learning rate scheduler
          scheduler0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer0, mode='min', factor=0.5, patience=5, verbose=True)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
          scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=5, verbose=True)
          scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.5, patience=5, verbose=True)
          scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='min', factor=0.5, patience=5, verbose=True)
          scheduler4 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer4, mode='min', factor=0.5, patience=5, verbose=True)
          scheduler5 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer5, mode='min', factor=0.5, patience=5, verbose=True)
          import time
          # Start the timer
          start_time = time.time()

          recalage_rigide(model, optimizer0, scheduler0, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, path)
          if step_pose:
            pose_optimization(step_pose_L_sc, model, optimizer, scheduler, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, path)

          shape_optimization(step_shape_L_chamfer, step_shape_L_P2S, step_shape_L_sc, model, optimizer1, optimizer2, scheduler1, scheduler2, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, PC, PC1, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, path)
          mesh_f = deformation_clothes(step_D_L_P2S, step_D_L_laplacien, step_D_L_normal, step_D_L_id, step_D_L_id_face, model, optimizer4, optimizer5, scheduler4, scheduler5, PC1, D, betas, expression, body_pose, global_orient, left_hand_pose, right_hand_pose, t_params, pose_prior, pointcloud, body_mapping, id_hand_twist, id_hand_coller, body_mapping1, path)

          # End the timer
          end_time = time.time()
          # Calculate the duration
          duration = end_time - start_time
          print(f"The process of optimization took {duration} seconds.")

          start_time = time.time()
          extract_texture_pifu(pifu_mesh,mesh_f,'/home/ext_fares_podform3d_com/test/data/smplx_uv.obj',path,uv_size=1024)
          # t_params,global_orient,left_hand_pose,right_hand_pose
          data={'gender':gender,'scale':s1_params,'beta':betas,'theta':body_pose,'expression':expression,'D':D,'t_params':t_params,'global_orient':global_orient,'left_hand_pose':left_hand_pose,'right_hand_pose':right_hand_pose}
          np.save(path+'/'+names[:-4]+ '_data.npy', data)

          end_time = time.time()
          # Calculate the duration
          duration = end_time - start_time
          print(f"The process of texture extraction took {duration} seconds.")

          start_time = time.time()
          path_texture=os.path.join(path, "texture.png")
          path_texture_out=os.path.join(path, "texture_interpolation.png")
          interpolated_image, mask_finale= inpaint_interpolation(path_texture, path_texture_out)
          end_time = time.time()
          # Calculate the duration
          duration = end_time - start_time
          print(f"The process of texture completion took {duration} seconds.")

          use_lama = False
          if use_lama:
            path_mask='/home/ext_fares_podform3d_com/test/data/combined_mask.png'
            path_lama = os.path.join(path, "texture_lama2.png")
            apply_lama(path_texture_out,path_mask, path_lama) 


if __name__ == "__main__":

  folder_name = "all_sub"

  MSE_loss = nn.MSELoss()

  body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,63, 64, 65,
                          20, 37, 38, 39, 66, 25, 26, 27,67, 28, 29, 30, 68, 34, 35, 36, 69,31, 32, 33, 70,
                          21, 52, 53, 54, 71, 40, 41, 42, 72,43, 44, 45, 73, 49, 50, 51, 74, 46,47, 48, 75] + list(np.arange(76, 127)) , dtype=np.int32)

  body_mapping1 = np.array(list(np.arange(76, 127)) , dtype=np.int32)

  if torch.cuda.is_available():
      device = torch.device("cuda:0")
      torch.cuda.set_device(device)
  else:
      device = torch.device("cpu")
  read_dictionary = np.load('/home/ext_fares_podform3d_com/test/data/sub_target.npy',allow_pickle='TRUE').item()
  idx_D=read_dictionary['idx']
  read_dictionary = np.load('/home/ext_fares_podform3d_com/test/data/not_penelized.npy',allow_pickle='TRUE').item()
  idx_not_penelized=read_dictionary['idx']
  read_dictionary = np.load('/home/ext_fares_podform3d_com/test/data/index_regularisation.npy',allow_pickle='TRUE').item()
  idx_regularisation=read_dictionary['idx']
  faces_regularisation=read_dictionary['faces']

  abs_path='/home/ext_fares_podform3d_com/test/data/recon/'

  abs_path_target = '/home/ext_fares_podform3d_com/test/data/target'

  abs_path_target_mesh = '/home/ext_fares_podform3d_com/test/data/target_mesh'

  out_path_recons='/home/ext_fares_podform3d_com/test/output_finale/'
  out_path_recons= os.path.join(out_path_recons, folder_name)
  if not os.path.exists(out_path_recons):
      # If it doesn't exist, create it
      os.mkdir(out_path_recons)

  save_path='/home/ext_fares_podform3d_com/test/output_finale_render/'
  if not os.path.exists(out_path_recons):
    os.mkdir(out_path_recons)
  save_path= os.path.join(save_path, folder_name)
  if not os.path.exists(save_path):
      # If it doesn't exist, create it
      os.mkdir(save_path)
# ***********************************************************************************

  # model_folder = osp.expanduser(osp.expandvars('models'))
  model_folder = '/home/ext_fares_podform3d_com/test/external/smplx/models'
  model_type = 'smplx'
  plot_joints = True
  use_face_contour = False
  gender = 'female'
  ext = 'npz'
  plotting_module = 'pyrender'
  num_betas =300
  num_expression_coeffs = 300
  sample_shape = True
  sample_expression = True

  # reconstruction //////////////////////////////////////////////////////////////////////////////////////

  main(abs_path, out_path_recons, model_folder, model_type, ext=ext,
        gender=gender, plot_joints=plot_joints,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        sample_shape=sample_shape,
        sample_expression=sample_expression,
        plotting_module=plotting_module,
        use_face_contour=use_face_contour)

    # render  //////////////////////////////////////////////////////////////////////////////////////
# ************************************************************************************************************

  # Check if the directory already exists
  out_path_rgb = os.path.join(save_path, "rgb")
  if not os.path.exists(out_path_rgb):
      # If it doesn't exist, create it
      os.mkdir(out_path_rgb)


  out_path_rgb_na = os.path.join(save_path, "rgb_na")
  if not os.path.exists(out_path_rgb_na):
      # If it doesn't exist, create it
      os.mkdir(out_path_rgb_na)
  # Check if the directory already exists
  out_path_silh = os.path.join(save_path, "silhouette")
  if not os.path.exists(out_path_silh):
      # If it doesn't exist, create it
      os.mkdir(out_path_silh)
  # Check if the directory already exists
  out_path_normals = os.path.join(save_path, "normals")
  if not os.path.exists(out_path_normals):
      # If it doesn't exist, create it
      os.mkdir(out_path_normals)


mesh=load_obj('/home/ext_fares_podform3d_com/test/data/smplx_uv.obj')

for (dirpath, dirnames,filenames) in walk(out_path_recons):
  for i,dir in enumerate(dirnames) :
    path_texture=os.path.join(out_path_recons, dir, "texture_interpolation.png")
    data_path=os.path.join(out_path_recons, dir, dir+"_data.npy")

    read_dictionary = np.load(data_path,allow_pickle='TRUE').item()
    betas=read_dictionary['beta'].detach().to(device)

    theta=read_dictionary['theta'].detach().to(device)

    D=read_dictionary['D'].detach().to(device)

    express=read_dictionary['expression'].detach().to(device)

    global_orient = read_dictionary['global_orient'].detach().to(device)

    t_params = read_dictionary['t_params'].detach().to(device)

    left_hand_pose = read_dictionary['left_hand_pose'].detach().to(device)

    right_hand_pose = read_dictionary['right_hand_pose'].detach().to(device)

    img = cv2.imread(path_texture)
    img=img[:,:,[2,1,0]]
    tex = torch.from_numpy(img / 255.)[None].to(device)
    texture = TexturesUV(maps=tex.to(torch.float32).to(device), faces_uvs=mesh[1][2].unsqueeze(0).to(device), verts_uvs=mesh[2][1].unsqueeze(0).to(torch.float32).to(device))

    model_folder = '/home/ext_fares_podform3d_com/test/external/smplx/models'
    model_type = 'smplx'
    plot_joints = True
    use_face_contour = False
    # gender = 'male'
    ext = 'npz'
    plotting_module = 'pyrender'
    num_betas = 300
    num_expression_coeffs = 300
    sample_shape = True
    sample_expression = True

    if dir[7:13]=='female':
      gender='female'
    else:
      gender='male'
    model = smplx.create(model_folder, model_type=model_type,
                        gender=gender, use_face_contour=use_face_contour,
                        num_betas=num_betas,
                        num_expression_coeffs=num_expression_coeffs,create_left_hand_pose=True,create_right_hand_pose=True,use_pca=False,
                        ext=ext).to(device)

    #output = model(right_hand_pose=right_hand_pose, left_hand_pose=left_hand_pose, reye_pose=reye_pose,leye_pose=leye_pose, jaw_pose=jaw_pose, betas=betas,body_pose=theta, expression=express, global_orient=global_orient, t_params=t_params, return_verts=True,D=D)
    output = model(betas=betas, expression=express,body_pose=theta,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose=right_hand_pose,
            return_verts=True)
    mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()+D).detach().cpu().numpy().squeeze(), faces=model.faces)

    # mesh_f = filter_humphrey(mesh_f, alpha=0.2, beta=0.3, iterations=50,
    #                                       laplacian_operator=None)

    verts=torch.tensor(mesh_f.vertices)

    mesh1=Meshes(verts=[verts.to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)

    # num_verts = verts.shape[1]
    # colors = torch.zeros(1, verts.shape[0], 3)

    # colors[0, :, 0] = torch.linspace(0.4, 0.8, verts.shape[0])  # Red channel
    # colors[0, :, 1] = torch.linspace(0.3, 0.9, verts.shape[0])  # Green channel
    # colors[0, :, 2] = torch.linspace(0.0, 1, verts.shape[0]) 
    # # Create a Textures object using the colors
    # texture = TexturesVertex(verts_features=colors.to(device))

    # # tex =torch.full_like(tex, 0.5).to(device)
    # # texture = TexturesUV(maps=tex.to(torch.float32).to(device), faces_uvs=mesh[1][2].unsqueeze(0).to(device), verts_uvs=mesh[2][1].unsqueeze(0).to(torch.float32).to(device))
    # mesh_na=Meshes(verts=[verts.to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)


    R, T = look_at_view_transform(1.8, 0, 0)
    cameras = OpenGLOrthographicCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=10,
        bin_size = None,
        max_faces_per_bin = None
    )


    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]],ambient_color=((1, 1, 1),),diffuse_color=((0.0, 0.0, 0.0),),specular_color=((0.0, 0.0, 0.0),))

    materials = Materials(
        device=device,
        specular_color=[[0.0, 0.0, 0.0]],
        shininess=5.0
    )


    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    images = renderer(mesh1)
    save_image(images[0,:,:,:3].permute((2,0,1)),out_path_rgb+'/'+dir[7:-4]+'.png')

    # images_na = renderer(mesh_na)
    # save_image(images_na[0,:,:,:3].permute((2,0,1)),out_path_rgb_na+'/'+dir[7:-4]+'_na.png')


    mesh1.textures = TexturesVertex(verts_features=(mesh1.verts_normals_padded() + 1.0) * 0.5)
    images = renderer(mesh1)
    save_image(images[0,:,:,:3].permute((2,0,1)),out_path_normals+'/'+dir[7:-4]+'.png')

    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=1e-9,
        faces_per_pixel=50,
        cull_backfaces=True,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(),
    )
    silhouette = renderer(mesh1, cameras=cameras).squeeze()
    image1 = transforms.ToPILImage()((silhouette > 0)[:,:,3].detach().cpu().numpy().astype('uint8')*255)
    image1.save(out_path_silh+'/'+dir[7:-4]+'.png')



# 2d evaluation  //////////////////////////////////////////////////////////////////////////////////////
# root_path=save_path
# in_path='/home/ext_fares_podform3d_com/test/data/target'
# abs_path_target
evaluation_pathes=['ours']
names=['ours']
types=['rgb','normals']

transform = transforms.ToTensor()

psnr = PeakSignalNoiseRatio()
lpips_alex = lpips.LPIPS(net='alex')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0) 
for i,type_select in enumerate(types) :
   print(type_select)
i=0
for i,type_select in enumerate(types) :
  print('evaluation de '+type_select+'--->')
  out_path_target = os.path.join(abs_path_target, type_select)
  out_path_target_silh = os.path.join(abs_path_target, 'silhouette')
  ssim_list=[]
  lpips_list=[]
  psnr_list=[]
  IOU_list=[]
  for (dirpath, dirnames, filenames) in walk(out_path_target):
    for dir in filenames:
      print(dir,'***********')
      im_path_target=os.path.join(out_path_target,dir)
      im_path_target_silh=os.path.join(out_path_target_silh,dir)

      im_path_predicted=os.path.join(os.path.join(save_path,type_select),dir)
      im_path_predicted_silh=os.path.join(os.path.join(save_path,'silhouette'),dir)

      IOU_path=os.path.join(save_path,'IOU')
      if not os.path.exists(IOU_path):
        os.mkdir(IOU_path)


      target_image = Image.open(im_path_target)
      target_image_silh = Image.open(im_path_target_silh)
      # target_image = target_image.resize((256,256))
      # target_image_silh = target_image_silh.resize((256,256))


      target_image_silh_torch = transform(target_image_silh)
      # save_image(target_image_silh_torch, '/content/target_image_silh_torch.png')

      mask_idx = np.where(np.asarray(target_image_silh_torch.squeeze())!=0)



      predicted_image = Image.open(im_path_predicted)
      predicted_image_silh = Image.open(im_path_predicted_silh)



      print(predicted_image.size)
      if predicted_image.size!=(1024,1024) :
        predicted_image = predicted_image.resize((1024,1024))
        predicted_image_silh = predicted_image_silh.resize((1024,1024))

      # predicted_image = predicted_image.resize((256,256))
      # predicted_image_silh = predicted_image_silh.resize((256,256))

      predicted_image_silh_torch = transform(predicted_image_silh)

      mask_idx = np.where(np.asarray(predicted_image_silh_torch.squeeze())!=0)


      IOU_score=calculate_iou(predicted_image_silh_torch,target_image_silh_torch)
      save_image(torch.abs(predicted_image_silh_torch-target_image_silh_torch), IOU_path+'/'+dir)


      target_tensor = transforms.ToTensor()(target_image).unsqueeze(0)/ 255.0
      predicted_tensor =  transforms.ToTensor()(predicted_image).unsqueeze(0)/ 255.0

      save_image(torch.abs(target_tensor-predicted_tensor)*255,  IOU_path+'/'+type_select+dir)

      psnr_score=psnr(predicted_tensor[0,:,mask_idx[0],mask_idx[1]], target_tensor[0,:,mask_idx[0],mask_idx[1]])
      # psnr_score1=psnr(predicted_tensor[0,:,mask_idx[0],mask_idx[1]]*255, target_tensor[0,:,mask_idx[0],mask_idx[1]]*255)

      x, y, w, h = cv2.boundingRect(np.column_stack(mask_idx))

      predicted_tensor = predicted_tensor[:,:,x:x + w,y:y + h]
      target_tensor = target_tensor[:,:,x:x + w,y:y + h]

      # psnr_score1=psnr(predicted_tensor*255, target_tensor*255)

      # ssim_score = ssim(predicted_tensor, target_tensor)
      ssim_score1 = ssim(predicted_tensor*255, target_tensor*255)
      lpips_score=lpips_alex.forward(predicted_tensor*255, target_tensor*255).detach()


      print(f"IOU: {IOU_score:.5f}")
      print(f"SSIM1: {ssim_score1:.4f}")
      print(f"lpips: {lpips_score.item():.5f}")
      print(f"psnr: {psnr_score:.5f}")
      # print(f"psnr1: {psnr_score1:.5f}")

      IOU_list.append(IOU_score)
      ssim_list.append(ssim_score1)
      lpips_list.append(lpips_score.item())
      psnr_list.append(psnr_score)


    if not os.path.exists(os.path.join( save_path, 'stats')):
        # Create the directory if it does not exist
        os.mkdir(os.path.join( save_path, 'stats'))
    csv_file = os.path.join( os.path.join( save_path, 'stats'), type_select+'_data.csv')
    print(csv_file)

    IOU_list=np.array(IOU_list)
    ssim_list=np.array(ssim_list)
    lpips_list=np.array(lpips_list)
    psnr_list=np.array(psnr_list)

    IOU_mean=IOU_list.mean()
    ssim_mean=ssim_list.mean()
    lpips_mean=lpips_list.mean()
    psnr_mean=psnr_list.mean()

    IOU_list=np.append(IOU_list,IOU_mean)
    ssim_list=np.append(ssim_list,ssim_mean)
    lpips_list=np.append(lpips_list,lpips_mean)
    psnr_list=np.append(psnr_list,psnr_mean)


    data = zip(IOU_list,ssim_list, lpips_list, psnr_list)

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['IOU','ssim', 'lpips', 'psnr'])
        writer.writerows(data)


    print(ssim_mean,lpips_mean,psnr_mean)


# 3d evaluation  //////////////////////////////////////////////////////////////////////////////////////

# path_in='/home/ext_fares_podform3d_com/test/data/target_mesh'
# path_pred=in_path
if not os.path.exists(os.path.join( save_path, 'stats_3d')):
    # Create the directory if it does not exist
    os.mkdir(os.path.join( save_path, 'stats_3d'))
loss_chamfer=[]
loss_normal=[]
loss_iou3d=[]

for (_, dirnames, _) in os.walk(abs_path_target_mesh):
  for j,dir in enumerate(dirnames):
    print(j)
    # if i==2:
    #   break
    path_mesh_t=os.path.join(abs_path_target_mesh, dir +'/frontmesh-f00001.obj')
    print(path_mesh_t)
    with open(os.path.join(abs_path_target_mesh, dir +'/gender.txt'), "r") as file:
      gender = file.read().replace(" ", "").replace("\n", "")
    print(gender)


    mesh_t = trimesh.load_mesh(path_mesh_t)
    #mesh_t.export('/content/xavatar.obj')
    mesh_t_torch=torch.tensor(mesh_t.vertices, dtype=torch.float32, requires_grad=True, device=device)


    mesh_f = trimesh.load_mesh(out_path_recons +'/result_'+gender+'-'+dir+'_512/smpl_final_clothes.obj')
    mesh_f.vertices[:,1]=mesh_f.vertices[:,1]-mesh_f.vertices[:,1].min(0)
    #mesh_f.export('/content/pifu.obj')
    mesh_f_torch=torch.tensor(mesh_f.vertices, dtype=torch.float32, requires_grad=True, device=device)

    translation = torch.zeros(3, requires_grad=True, device=device)
    rotation = torch.eye(3, requires_grad=True, device=device)
    scale = torch.tensor(1.0, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([translation, scale], lr=0.001)

    # Optimize the transformation parameters
    for i in range(200):
        # Transform one of the point clouds
        transformed_y = scale * torch.matmul(mesh_f_torch, rotation) + translation

        # Compute the Chamfer loss
        loss = chamfer_distance(mesh_t_torch.unsqueeze(0), transformed_y.unsqueeze(0))[0]


        # Backpropagate the gradients
        loss.backward()
        # print(loss*1e3)

        # Update the transformation parameters
        optimizer.step()
        optimizer.zero_grad()


    # compare_meshes()
    mesh_f_tensor=Meshes(verts=[(transformed_y.detach().to(device))], faces=[(torch.tensor(mesh_f.faces.astype(np.int32),dtype=torch.int64)).to(device)])
    mesh_t_tensor=Meshes(verts=[(mesh_t_torch.detach().to(device))], faces=[(torch.tensor(mesh_t.faces.astype(np.int32),dtype=torch.int64)).to(device)])

    # metrics=compare_meshes(mesh_f_tensor, mesh_t_tensor, num_samples=1)
    # print(mesh_normal_consistency(mesh_f_tensor),'****')
    # print(loss.item())
    loss_chamfer.append(loss.item()*1e3)
    #loss_normal.append(mesh_normal_consistency(mesh_f_tensor).item())

    mesh_f.vertices=transformed_y.detach().cpu().numpy()
    loss_normal.append(normal_consistency_metric_nn(mesh_t,mesh_f))

    # loss_iou3d.append(calculate_iou(mesh_f, mesh_t))
    # print(loss_iou3d[-1])



loss_normal=np.array(loss_normal)
loss_chamfer=np.array(loss_chamfer)
# print(loss_normal,loss_chamfer)

loss_normal_m=loss_normal.mean()
loss_chamfer_m=loss_chamfer.mean()

loss_normal=np.append(loss_normal,loss_normal_m)
loss_chamfer=np.append(loss_chamfer,loss_chamfer_m)
print(loss_normal,'****',loss_chamfer)


csv_file = os.path.join( os.path.join( save_path, 'stats_3d'),'3d_data.csv')
print(csv_file)

data = zip(loss_normal,loss_chamfer )

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['loss_normal','loss_chamfer'])
    writer.writerows(data)