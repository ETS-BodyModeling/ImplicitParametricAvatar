from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures import Meshes
from pytorch3d.structures import Pointclouds
import numpy as np
# from sklearn.neighbors import NearestNeighbors
from dgl.geometry import farthest_point_sampler
import os.path as osp
import argparse
import trimesh
import numpy as np
import torch
import smplx
from os import walk
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_face_distance,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

import numpy as np
import torch
import smplx
import torch.nn as nn
from functions import *
from util_texture import inpaint_interpolation, apply_lama
import os
import os.path as osp
from os import walk

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

    i=0

    id_hand_twist=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 39, 42]
    id_hand_coller=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 40, 43]
    for (dirpath, dirnames, filenames) in walk(abs_path):

      for names in filenames:
        if names.endswith(".obj"):
          print(names[7:13])
          if names[7:13]=='female':
            gender='female'
          else:
            gender='male'
          i=i+1
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
          optimizer4 = torch.optim.Adam([D], lr=5*1e-5)
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

          print('RECALAGE RIGIDE///////////////////////////////////////////////////////////////////////////////////////////////')

          patience = 10  # Number of epochs to wait for loss improvement
          min_delta = 0.0001  # Minimum change in loss to be considered as an improvement
          best_loss = float('inf')
          epochs_without_improvement = 0

          for i in range(100):
            optimizer0.zero_grad()
            # print(t_params)
            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose=right_hand_pose,
                      return_verts=True)

            loss=torch.norm(betas[0,1:])/5+MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,15,16,21,24]],pose_prior[:,[0,15,16,21,24]]) + MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,22]].mean(1))
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



          print('3D pose optimization////////////////////////////////////////////////////////////////////////////////////////////////////////')

          for i in range(150):
            # print(left_hand_pose)
            optimizer.zero_grad()
            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)
            # loss=10*torch.norm(body_pose.reshape((21,3))[[2,5,8,9,10,11,12,13,14],:]) +MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]])+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
            # loss= MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]])+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
            # loss= MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]],pose_prior[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]])
            loss=10*torch.norm(body_pose.reshape((21,3))[[2,5,8,9,10,11,12,13,14],:]) +MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]])+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 2*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
            
            print(i, loss)
            loss.backward()
            optimizer.step()

            scheduler.step(loss)


          mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
          mesh_f.export(path+ '/smpl_med.obj')
          print(path+ "/smpl_med.obj")


          patience = 10  # Number of epochs to wait for loss improvement
          min_delta = 0.001  # Minimum change in loss to be considered as an improvement
          best_loss = float('inf')
          epochs_without_improvement = 0

          print('3D pose and shape optim Shamfer distance///////////////////////////////////////////////////////////////////////////////////////////////')
          for i in range(100):
            optimizer1.zero_grad()
            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)

            smpl_mesh=Meshes(verts=[output.vertices.squeeze().to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])

            loss =  chamfer_distance(output.vertices.to(device),pointcloud)[0] +  MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]])  + 10*MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]],pose_prior[:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]])+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
            # loss = 1e4*loss_face+ chamfer_distance(output.vertices.to(device),y)[0] +  MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]],pose_prior1[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,24]])  + 10*MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]],pose_prior1[:,[0,2,3,4,5,6,7,9,10,12,13,15,16,17,18]])+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior1[:,[22,23,24]].mean(1))+ MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior1[:,[19,20,21]].mean(1))
            print(i, loss)
            loss.backward()

            optimizer1.step()

          mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
          mesh_f.export(path+ '/smpl_med1.obj')

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



          patience = 10  # Number of epochs to wait for loss improvement
          min_delta = 1  # Minimum change in loss to be considered as an improvement
          best_loss = float('inf')
          epochs_without_improvement = 0

          print('3D pose and shape optim (Shamfer distance,point2surface loss)///////////////////////////////////////////////////////////////////////////////////////////////')
          for i in range(100):
            optimizer2.zero_grad()
            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)

            smpl_mesh=Meshes(verts=[output.vertices.squeeze().to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
            # PC=Pointclouds(points=list(pointcloud))
            losss=knn_loss1(smpl_mesh,PC)

            hand_penal=loss_with_constraint( left_hand_pose, -0.8, 0.5, 1e6) + loss_with_constraint( right_hand_pose, -0.8, 0.5, 1e6) +torch.norm(left_hand_pose[0,id_hand_twist])*1e3 +torch.norm(left_hand_pose[0,id_hand_coller])*1e2  + torch.norm(right_hand_pose[0,id_hand_twist])*1e3 + torch.norm(right_hand_pose[0,id_hand_coller])*1e2

            losss1=10*chamfer_distance(output.vertices.to(device),pointcloud)[0]
            loss_face=1e4*MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:])
            loss=  hand_penal +loss_face+ losss1 + losss + 1e4* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]) + 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
            print(i,loss_face,losss,losss1,hand_penal, loss)
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

          print('3D pose and shape optim point2surface loss)///////////////////////////////////////////////////////////////////////////////////////////////')
          for i in range(100):
            optimizer2.zero_grad()

            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)

            smpl_mesh=Meshes(verts=[output.vertices.squeeze().to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
            # PC=Pointclouds(points=list(pointcloud))
            losss=knn_loss1(smpl_mesh,PC1)

            loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:]) 
            hand_penal=loss_with_constraint( left_hand_pose, -0.8, 0.5, 1e7) + loss_with_constraint( right_hand_pose, -0.8, 0.5, 1e7) + torch.norm(right_hand_pose[0,id_hand_twist])*1e2 +torch.norm(left_hand_pose[0,id_hand_coller])*1e1  + torch.norm(right_hand_pose[0,id_hand_twist])*1e2 + torch.norm(right_hand_pose[0,id_hand_coller])*1e1
            loss= 1e3 * hand_penal +  1e4*loss_face + losss + 1e4* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,]],pose_prior[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]) + 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior[:,[22,23,24]].mean(1))+ 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior[:,[19,20,21]].mean(1))
            
            # losss1=10*chamfer_distance(output.vertices.to(device),y)[0]
            # loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior1[:,44:])
            # loss=  1e4*loss_face+ losss1 + losss + 1e4* MSE_loss(output.joints.to(device)[:,body_mapping][:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,]],pose_prior1[:,[0,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]]) + 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[22,23,24]].mean(1),pose_prior1[:,[22,23,24]].mean(1))+ 1e3*MSE_loss(output.joints.to(device)[:,body_mapping][:,[19,20,21]].mean(1),pose_prior1[:,[19,20,21]].mean(1))
            
            print(i,loss_face,losss, loss)
            loss.backward()
            optimizer2.step()
          mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(),faces=model.faces)
          mesh_f.export(path+ '/smpl_final1.obj')




          print('D clothes deformation///////////////////////////////////////////////////////////////////////////////////////////////')
          for i in range(200):
            optimizer4.zero_grad()

            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)
            smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
            smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()+D[idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])

            losss=knn_loss2(smpl_mesh,PC1)
            # loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:])
            # loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior1[:,86:])
            loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
            loss_normal=mesh_normal_consistency(smpl_mesh_reg)
            loss=2*losss  +    1e4*loss_laplacian+   1e4*loss_normal +  1e6 * torch.norm(torch.norm(D[idx_D], dim=1)) + torch.norm(torch.norm(D, dim=1))
            print(i,losss,loss)
            loss.backward()
            optimizer4.step()
            scheduler4.step(losss)

          mesh_f.export(path+ '/smpl_final_clothes_0.obj')

          patience = 10  # Number of epochs to wait for loss improvement
          min_delta = 0.1  # Minimum change in loss to be considered as an improvement
          best_loss = float('inf')
          epochs_without_improvement = 0
          print('D clothes deformation 1///////////////////////////////////////////////////////////////////////////////////////////////')
          for i in range(1000):
            optimizer5.zero_grad()

            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)
            smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
            smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()+D[idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])

            losss=knn_loss2(smpl_mesh,PC1)

            loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
            loss_normal=mesh_normal_consistency(smpl_mesh_reg)
            loss=2*losss  +    1e1*loss_laplacian+   1e2*loss_normal +  1e6 * torch.norm(torch.norm(D[idx_D], dim=1)) + torch.norm(torch.norm(D, dim=1))
            print(i,losss,1e2*loss_laplacian, 1e2*loss_normal,torch.norm(torch.norm(D, dim=1)))
            loss.backward()
            optimizer5.step()
            if best_loss - losss > min_delta:
                  best_loss = losss
                  epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler5.step(losss)

            # Check if training should stop
            if epochs_without_improvement == patience:
                print("Early stopping. No improvement in loss.")
                break


          print('D clothes deformation 2///////////////////////////////////////////////////////////////////////////////////////////////')
          for i in range(100):
            optimizer5.zero_grad()

            output = model(betas=betas, expression=expression,body_pose=body_pose,transl=t_params,global_orient=global_orient,left_hand_pose=left_hand_pose,right_hand_pose =right_hand_pose ,
                      return_verts=True)
            smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
            smpl_mesh_reg=Meshes(verts=[(output.vertices[0,idx_regularisation].squeeze()+D[idx_regularisation].squeeze()).to(device)], faces=[(torch.tensor(faces_regularisation,dtype=torch.int32)).to(device)])

            losss=knn_loss2(smpl_mesh,PC1)
            # loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior[:,44:])
            # loss_face=MSE_loss(output.joints.to(device)[:,body_mapping1],  pose_prior1[:,86:])
            loss_laplacian = mesh_laplacian_smoothing(smpl_mesh, method="uniform")
            loss_normal=mesh_normal_consistency(smpl_mesh_reg)
            loss=2.5*losss  +    1e1*loss_laplacian+   1e2*loss_normal +  1e6 * torch.norm(torch.norm(D[idx_D], dim=1)) + torch.norm(torch.norm(D, dim=1))
            print(i,losss,1e2*loss_laplacian, 1e2*loss_normal,torch.norm(torch.norm(D, dim=1)))
            loss.backward()
            optimizer5.step()
            if best_loss - losss > min_delta:
                  best_loss = losss
                  epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler5.step(losss)

            # Check if training should stop
            if epochs_without_improvement == patience:
                print("Early stopping. No improvement in loss.")
                break
          mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()+D).detach().cpu().numpy(),faces=model.faces)

          # End the timer
          end_time = time.time()
          # Calculate the duration
          duration = end_time - start_time
          print(f"The process of optimization took {duration} seconds.")

          mesh_f.export(path+ '/smpl_final_clothes.obj')

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
  out_path='/home/ext_fares_podform3d_com/test/output/base/'
  if not os.path.exists(out_path):
    os.mkdir(out_path)


  # model_folder = osp.expanduser(osp.expandvars('models'))
  model_folder = '/home/ext_fares_podform3d_com/test/external/smplx/models'
  print(model_folder)
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

  main(abs_path, out_path, model_folder, model_type, ext=ext,
        gender=gender, plot_joints=plot_joints,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        sample_shape=sample_shape,
        sample_expression=sample_expression,
        plotting_module=plotting_module,
        use_face_contour=use_face_contour)