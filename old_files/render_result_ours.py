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
from trimesh.smoothing import filter_humphrey
from torchvision.transforms import ToTensor, transforms
from pytorch3d.transforms import Rotate
from pytorch3d.renderer import TexturesVertex
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
    FoVPerspectiveCameras,
    OpenGLOrthographicCameras,
    PerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    TexturesUV
)
import os 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
#device = torch.device("cpu")
save_path='/home/ext_fares_podform3d_com/test/output_render_base1/ours'
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



path='/home/ext_fares_podform3d_com/test/output/base_1'
mesh=load_obj('/home/ext_fares_podform3d_com/test/data/smplx_uv.obj')

for (dirpath, dirnames,filenames) in walk(path):
  for i,dir in enumerate(dirnames) :
    path_texture=os.path.join(path, dir, "texture_interpolation.png")
    data_path=os.path.join(path, dir, dir+"_data.npy")

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
    #smpl_mesh=Meshes(verts=[(output.vertices.squeeze()+D).to(device)], faces=[(torch.tensor(model.faces.astype(np.float64),dtype=torch.int32)).to(device)])
    mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()+D).detach().cpu().numpy().squeeze(), faces=model.faces)

    # mesh_f = filter_humphrey(mesh_f, alpha=0.2, beta=0.3, iterations=50,
    #                                       laplacian_operator=None)

    verts=torch.tensor(mesh_f.vertices)

    mesh1=Meshes(verts=[verts.to(torch.float32).to(device)], faces=[(torch.tensor(model.faces.astype(np.int32),dtype=torch.int64)).to(device)], textures=texture)


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
