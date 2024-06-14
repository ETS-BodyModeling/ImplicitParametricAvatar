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
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV
)
import cv2
import numpy as np
from dgl.geometry import farthest_point_sampler
import os.path as osp
import argparse
import trimesh
import numpy as np
import torch
import smplx
from os import walk
import cv2
from pytorch3d.transforms import RotateAxisAngle
from torchvision.utils import save_image
import shutil
import math
import os
from trimesh.smoothing import filter_humphrey
# from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.transforms import Rotate

save_path='/home/ext_fares_podform3d_com/test/outpur_render'
# Check if the directory already exists
out_path_rgb = os.path.join(save_path, "rgb")
if not os.path.exists(out_path_rgb):
    # If it doesn't exist, create it
    os.mkdir(out_path_rgb)
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



path='/home/ext_fares_podform3d_com/test/output/base'
mesh=load_obj('/home/ext_fares_podform3d_com/test/data/smplx_uv.obj')

for (dirpath, dirnames, _) in walk(path):
  for i,dir in enumerate(dirnames) :

    path_texture=os.path.join(path, dir, "texture_interpolation.png")
    data_path=os.path.join(path, dir, dir+"_data.npy")
    save_path=path+'/'+dir

    read_dictionary = np.load(data_path,allow_pickle='TRUE').item()
    beta=read_dictionary['beta']
    theta=read_dictionary['theta']
    D=read_dictionary['D']
    express=read_dictionary['expression']
    
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
                          num_expression_coeffs=num_expression_coeffs,
                          ext=ext)

    betas = torch.zeros((1, model.num_betas), dtype=torch.float32)
    betas[0,:]=beta.detach()[0,:]

    expression = torch.zeros(
        [1, model.num_expression_coeffs], dtype=torch.float32)


    output = model(betas=betas,return_verts=True,D=D)
    # output = model(betas=betas,body_pose=theta, expression=express,return_verts=True,D=D)
    
    mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export('/content/1.obj')
    mesh_f = filter_humphrey(mesh_f, alpha=0.2, beta=0.3, iterations=50,
                                          laplacian_operator=None)
    # mesh_f.export('/content/2.obj')
    verts=torch.tensor(mesh_f.vertices)

    mesh1=Meshes(verts=[verts.to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)

    # mesh1=Meshes(verts=[(output.vertices.squeeze()).to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)

    for val in range(4):
      print(val)
      theta1 = torch.tensor([math.pi*val/2]) 


      R_y = torch.tensor([
  [torch.cos(theta1), 0, torch.sin(theta1)],
  [0, 1, 0],
  [-torch.sin(theta1), 0, torch.cos(theta1)]
  ])

      new_verts = torch.matmul(mesh1.verts_packed(), R_y.to(mesh1.device))
      mesh2=Meshes(verts=[(new_verts.to(device))], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)
      
      
      R, T = look_at_view_transform(2, 20, 0, at=((0,-0.5, 0),))

      cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

      raster_settings = RasterizationSettings(
          image_size=1024, 
          blur_radius=0.0, 
          faces_per_pixel=30, 
      )

      lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]],ambient_color=((0.7, 0.7, 0.7),),diffuse_color=((0.3, 0.3, 0.3),),specular_color=((0.1, 0.1, 0.1),))

      materials = Materials(
          device=device,
          specular_color=[[0.1, 0.1, 0.1]],
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

      images = renderer(mesh2, lights=lights, materials=materials)
      print(save_path+'/{:03}.png'.format(val))

      save_image(images[0,:,:,:3].permute((2,0,1)),save_path+'/{:03}_1.png'.format(val))



# //////////////////////////////////////////////

    output = model(betas=betas,body_pose=theta, expression=express,return_verts=True,D=D)
    
    mesh_f = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces)
    # mesh_f.export('/content/1.obj')
    mesh_f = filter_humphrey(mesh_f, alpha=0.2, beta=0.3, iterations=50,
                                          laplacian_operator=None)
    # mesh_f.export('/content/2.obj')
    verts=torch.tensor(mesh_f.vertices)

    mesh1=Meshes(verts=[verts.to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)

    # mesh1=Meshes(verts=[(output.vertices.squeeze()).to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)

    for val in range(4):
      print(val)
      theta1 = torch.tensor([math.pi*val/2]) 


      R_y = torch.tensor([
  [torch.cos(theta1), 0, torch.sin(theta1)],
  [0, 1, 0],
  [-torch.sin(theta1), 0, torch.cos(theta1)]
  ])

      new_verts = torch.matmul(mesh1.verts_packed(), R_y.to(mesh1.device))
      mesh2=Meshes(verts=[(new_verts.to(device))], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)
      
      
      R, T = look_at_view_transform(2, 20, 0, at=((0,-0.5, 0),))

      cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

      raster_settings = RasterizationSettings(
          image_size=1024, 
          blur_radius=0.0, 
          faces_per_pixel=30, 
      )

      lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]],ambient_color=((0.7, 0.7, 0.7),),diffuse_color=((0.3, 0.3, 0.3),),specular_color=((0.1, 0.1, 0.1),))

      materials = Materials(
          device=device,
          specular_color=[[0.1, 0.1, 0.1]],
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

      images = renderer(mesh2, lights=lights, materials=materials)
      print(save_path+'/{:03}.png'.format(val))

      save_image(images[0,:,:,:3].permute((2,0,1)),save_path+'/posed_{:03}_1.png'.format(val))