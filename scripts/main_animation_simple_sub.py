
import numpy as np
import trimesh
from pathlib import Path
import sys
import torch
from submodules.smplx import smplx
from os import walk
import os
import cv2
import argparse

from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as transforms


from pytorch3d.io import load_obj
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


_ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_ROOT_DIR))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
import math


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Render SMPLX models with textures")
    parser.add_argument('--output_path', type=str, default='output/recons', help='Output directory for reconstructed mesh')
    parser.add_argument('--data_path', type=str, default='data', help='Directory for target data')
    parser.add_argument('--output_path_render', type=str, default='output/output_animation', help='Output directory for reconstructed mesh')
    return parser.parse_args(args)


if __name__ == "__main__":

  root_path = _ROOT_DIR.parent
  args = parse_args()

  gender = "female"
 


  save_path = os.path.join(root_path, args.output_path_render)
  out_path_recons = os.path.join(root_path, args.output_path)

  if not os.path.exists(save_path):
    os.mkdir(save_path)
#   save_path= os.path.join(save_path)
#   if not os.path.exists(save_path):
#       os.mkdir(save_path)



#mesh=load_obj(os.path.join(root_path, args.data_path, 'smplx_uv_simple.obj'))
mesh=load_obj(os.path.join(root_path, args.data_path, 'smplx_uv_simple_sub.obj'))
mesh_simple = trimesh.load_mesh('/home/fares/ImplicitParametricAvatar/data/smplx_uv_simple.obj', process=False, maintain_order=True)
mesh_simple_sub = trimesh.load_mesh('/home/fares/ImplicitParametricAvatar/data/smplx_uv_simple_sub.obj', process=False, maintain_order=True)

for (dirpath, dirnames,filenames) in walk(out_path_recons):
    for i,dir in enumerate(dirnames) :
        print(dir)
        if i == 14:
            


            path_texture=os.path.join(out_path_recons, dir, "texture_interpolation_simple_lama.png")
            data_path=os.path.join(out_path_recons, dir, dir+"_data.npy")

            out_path_rgb = os.path.join(save_path, dir)
            if not os.path.exists(out_path_rgb):
                os.mkdir(out_path_rgb)

            read_dictionary = np.load(data_path,allow_pickle='TRUE').item()
            betas=read_dictionary['beta'].detach().to(device)
            gender=read_dictionary['gender']

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

            model_folder = os.path.join(root_path, 'models')
            model_type = 'smplx'
            plot_joints = True
            use_face_contour = False
            ext = 'npz'
            plotting_module = 'pyrender'
            num_betas = 300
            num_expression_coeffs = 300
            sample_shape = True
            sample_expression = True


            model = smplx.create(model_folder, model_type=model_type,
                                gender=gender, use_face_contour=use_face_contour,
                                num_betas=num_betas,
                                num_expression_coeffs=num_expression_coeffs,create_left_hand_pose=True,create_right_hand_pose=True,use_pca=False,
                                ext=ext).to(device)

            with np.load(os.path.join(root_path, 'data', 'motions', 'mosh', 'D5_-_Random_Stuff_2_stageii.npz'), allow_pickle=True) as data:
                if 'mocap_frame_rate' in data:
                    fps = data['mocap_frame_rate']
                else:
                    fps = 120

                body_pose=data['pose_body']
                num_data=body_pose.shape[0]
                body_pose=torch.tensor(body_pose, dtype=torch.float32).reshape((num_data,1,21*3)).to(device)

                left_hand_pose=data['pose_hand'][:,:45]
                left_hand_pose=torch.tensor(left_hand_pose, dtype=torch.float32).reshape((num_data,1,15*3)).to(device)

                right_hand_pose=data['pose_hand'][:,45:]
                right_hand_pose=torch.tensor(right_hand_pose, dtype=torch.float32).reshape((num_data,1,15*3)).to(device)

                global_orient=data['root_orient']
                global_orient=torch.tensor(global_orient, dtype=torch.float32).reshape((num_data,1,1*3)).to(device)


                transl=data['trans']
                transl=torch.tensor(transl, dtype=torch.float32).reshape((num_data,1,1*3)).to(device)

                for kkk in range(num_data):
                    print(kkk)
                    output = model(global_orient=global_orient[kkk],betas=betas, expression=express,body_pose=body_pose[kkk],left_hand_pose=left_hand_pose[kkk],right_hand_pose=right_hand_pose[kkk],return_verts=True,D=D, faces_sub=mesh_simple.faces)
                    # mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy().squeeze(), faces=model.faces, process=False, maintain_order=True)
                    mesh_simple = trimesh.load_mesh('/home/fares/ImplicitParametricAvatar/data/smplx_uv_simple.obj', process=False, maintain_order=True)
                    mesh_f = trimesh.Trimesh(vertices=(output.vertices.squeeze()).detach().cpu().numpy().squeeze(), faces=mesh_simple.faces, process=False, maintain_order=True)
                    verts=torch.tensor(mesh_f.vertices)

                #     theta1 = torch.tensor([math.pi/2])
                #     R_y = torch.tensor([
                # [0, 1, 0],
                # [torch.cos(theta1), 0, torch.sin(theta1)],
                # [-torch.sin(theta1), 0, torch.cos(theta1)]
                
                
                # ])

                #     verts = torch.matmul(verts.float().to(device), R_y.float().to(device))


                    mesh1=Meshes(verts=[verts.to(torch.float32).to(device)], faces=[(torch.tensor(mesh_simple_sub.faces,dtype=torch.int64)).to(device)], textures=texture)
                    theta = torch.tensor([math.pi/2]) # rotate .5 radians in the Y-axis
                    theta1 = torch.tensor([math.pi])

                    R_y = torch.tensor([
                [torch.cos(theta1), 0, torch.sin(theta1)],
                [0, 1, 0],
                [-torch.sin(theta1), 0, torch.cos(theta1)]
                ])
                    R_z = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
                ])
                    R = torch.tensor([
                        [1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta), torch.cos(theta)]
                    ])

                    new_verts = torch.matmul(mesh1.verts_packed(), R.to(mesh1.device))#.unsqueeze(0)
                    new_verts = torch.matmul(new_verts, R_y.to(mesh1.device))
                    mesh1=Meshes(verts=[(new_verts.to(device))], faces=[(torch.tensor(mesh_simple_sub.faces,dtype=torch.int64)).to(device)], textures=texture)
                    R, T = look_at_view_transform(1.5, 1, 3, at=((0,0, 0),))
                    #R, T = look_at_view_transform(1.8, 0, 0)
                    cameras = OpenGLOrthographicCameras(device=device, R=R, T=T)
                    raster_settings = RasterizationSettings(
                        image_size=1024,
                        blur_radius=1e-7, 
                        faces_per_pixel=120,
                        cull_backfaces=False,
                        bin_size = 0,
                        max_faces_per_bin = None,
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
                    # images = torch.rot90(images, k=1, dims=(1, 2)) 
                    save_image(images[0,:,:,:3].permute((2,0,1)),out_path_rgb+'/' + '{:04}.png'.format(kkk))

                # Set the path to the directory containing the images

                # Get a list of all the image file names in the directory
                image_names = os.listdir(out_path_rgb)

                # Sort the list of file names in ascending order
                image_names.sort()

                # Set the frame rate of the resulting video
                frame_rate =120

                # Set the size of the video frames to the size of the first image
                first_image = cv2.imread(os.path.join(out_path_rgb, image_names[0]))
                frame_size = (first_image.shape[1], first_image.shape[0])

                # Create a VideoWriter object to write the video
                output_video = cv2.VideoWriter(out_path_rgb + '/animation.mp4', cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, frame_size)

                # Loop through each image and add it to the video
                for image_name in image_names:
                    print(os.path.join(out_path_rgb, image_name))
                    image = cv2.imread(os.path.join(out_path_rgb, image_name))
                    output_video.write(image)

                # Release the VideoWriter object
                output_video.release()
