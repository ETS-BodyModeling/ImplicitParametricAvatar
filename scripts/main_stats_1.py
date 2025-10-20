from pytorch3d.structures import Meshes
import numpy as np
import trimesh
import torch
import os
from os import walk
import cv2
from torchvision.utils import save_image
from torchvision.transforms import transforms
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from pytorch3d.structures import Meshes

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import lpips
import os
from PIL import Image
from torchvision import transforms
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from pytorch3d.loss import (
    chamfer_distance,
)
import os
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import csv
from pathlib import Path
import sys
import argparse
# this is a stat file to evaluate the results of the 3d and 2d reconstruction
_ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_ROOT_DIR))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

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

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Render SMPLX models with textures")
    parser.add_argument('--output_path', type=str, default='output_simple/recons', help='Output directory for reconstructed mesh')
    parser.add_argument('--data_path', type=str, default='data', help='Directory for target data')
    parser.add_argument('--output_path_render', type=str, default='output_simple/output_render', help='Output directory for reconstructed mesh')
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args()

    root_path = _ROOT_DIR.parent
    bs_path_target = os.path.join(root_path, args.data_path, 'target')
    abs_path_target = os.path.join(root_path, args.data_path, 'target')
    abs_path_target_mesh = os.path.join(root_path, args.data_path, 'target_mesh')
    out_path_recons = os.path.join(root_path, args.output_path)

    save_path = os.path.join(root_path,args.output_path_render)
    save_path= os.path.join(save_path)

    

    # 2d evaluation  //////////////////////////////////////////////////////////////////////////////////////
    evaluation_pathes=['ours']
    names=['ours']
    types=['normals', 'rgb']

    transform = transforms.ToTensor()

    psnr = PeakSignalNoiseRatio()
    lpips_alex = lpips.LPIPS(net='alex')
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0) 

    mean_data=[]

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
                # if dir != "female-29.png":
                #     continue
                # print(dir,'***********')
                im_path_target=os.path.join(out_path_target,dir)
                im_path_target_silh=os.path.join(out_path_target_silh,dir)

                im_path_predicted=os.path.join(os.path.join(save_path,type_select),dir)
                im_path_predicted_silh=os.path.join(os.path.join(save_path,'silhouette'),dir)

                IOU_path=os.path.join(save_path,'IOU')
                if not os.path.exists(IOU_path):
                    os.mkdir(IOU_path)


                target_image = Image.open(im_path_target)
                target_image_silh = Image.open(im_path_target_silh)

                target_image_silh_torch = transform(target_image_silh)

                mask_idx = np.where(np.asarray(target_image_silh_torch.squeeze())!=0)

                predicted_image = Image.open(im_path_predicted)
                predicted_image_silh = Image.open(im_path_predicted_silh)



                # print(predicted_image.size)
                if predicted_image.size!=(1024,1024) :
                    predicted_image = predicted_image.resize((1024,1024))
                    predicted_image_silh = predicted_image_silh.resize((1024,1024))


                predicted_image_silh_torch = transform(predicted_image_silh)

                mask_idx = np.where(np.asarray(predicted_image_silh_torch.squeeze())!=0)


                IOU_score=calculate_iou(predicted_image_silh_torch,target_image_silh_torch)
                save_image(torch.abs(predicted_image_silh_torch-target_image_silh_torch), IOU_path+'/'+dir)


                target_tensor = transforms.ToTensor()(target_image).unsqueeze(0)/ 255.0
                predicted_tensor =  transforms.ToTensor()(predicted_image).unsqueeze(0)/ 255.0

                save_image(torch.abs(target_tensor-predicted_tensor)*255,  IOU_path+'/'+type_select+dir)

                psnr_score=psnr(predicted_tensor[0,:,mask_idx[0],mask_idx[1]], target_tensor[0,:,mask_idx[0],mask_idx[1]])

                x, y, w, h = cv2.boundingRect(np.column_stack(mask_idx))

                predicted_tensor = predicted_tensor[:,:,x:x + w,y:y + h]
                target_tensor = target_tensor[:,:,x:x + w,y:y + h]

                ssim_score1 = ssim(predicted_tensor*255, target_tensor*255)
                lpips_score=lpips_alex.forward(predicted_tensor*255, target_tensor*255).detach()


                # print(f"IOU: {IOU_score:.5f}")
                # print(f"SSIM1: {ssim_score1:.4f}")
                # print(f"lpips: {lpips_score.item():.5f}")
                # print(f"psnr: {psnr_score:.5f}")

                IOU_list.append(IOU_score)
                ssim_list.append(ssim_score1)
                lpips_list.append(lpips_score.item())
                psnr_list.append(psnr_score)



            # Create the 'stats' directory in the parent of save_path (parent of --output_path_render)
            stats_parent = os.path.dirname(save_path)
            stats_dir = os.path.join(stats_parent, 'stats')
            if not os.path.exists(stats_dir):
                os.mkdir(stats_dir)
            csv_file = os.path.join(stats_dir, type_select+'_data.csv')
            # print(csv_file)

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

            mean_data.append([ssim_mean,lpips_mean,psnr_mean, IOU_mean])


            data = zip(IOU_list,ssim_list, lpips_list, psnr_list)

            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['IOU','ssim', 'lpips', 'psnr'])
                writer.writerows(data)


            # print(ssim_mean,lpips_mean,psnr_mean)




    # 3d evaluation  //////////////////////////////////////////////////////////////////////////////////////
    data_3d_exists = True
    if abs_path_target_mesh is None or not os.path.exists(abs_path_target_mesh):
        print(f"Error: Target mesh data directory does not exist: {abs_path_target_mesh}")
        data_3d_exists = False

    if data_3d_exists:  

        moyenne = [0.907,	0.805,	0.870,	0.125,	20.827,	0.900,	0.073,	23.229,	0.976]
        std = [0.00460,	0.00045,	0.00030,	0.00053,	0.02335,	0.00028,	0.00056,	0.06945,	0.00018]
        if not os.path.exists(os.path.join( stats_parent, 'stats_3d')):
            # Create the directory if it does not exist
            os.mkdir(os.path.join( stats_parent, 'stats_3d'))
        loss_chamfer=[]
        loss_normal=[]
        loss_iou3d=[]

        for (_, dirnames, _) in os.walk(abs_path_target_mesh):
            for j,dir in enumerate(dirnames):
                print(j)
                path_mesh_t=os.path.join(abs_path_target_mesh, dir +'/frontmesh-f00001.obj')
                print(path_mesh_t)
                with open(os.path.join(abs_path_target_mesh, dir +'/gender.txt'), "r") as file:
                    gender = file.read().replace(" ", "").replace("\n", "")
                    print(gender)


                mesh_t = trimesh.load_mesh(path_mesh_t)
                mesh_t_torch=torch.tensor(mesh_t.vertices, dtype=torch.float32, requires_grad=True, device=device)


                mesh_f = trimesh.load_mesh(out_path_recons +'/result_'+gender+'-'+dir+'_512/smpl_final_clothes_simple.obj')
                mesh_f.vertices[:,1]=mesh_f.vertices[:,1]-mesh_f.vertices[:,1].min(0)

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

                loss_chamfer.append(loss.item()*1e3)

                mesh_f.vertices=transformed_y.detach().cpu().numpy()
                loss_normal.append(normal_consistency_metric_nn(mesh_t,mesh_f))




        loss_normal=np.array(loss_normal)
        loss_chamfer=np.array(loss_chamfer)


        loss_normal_m=loss_normal.mean()
        loss_chamfer_m=loss_chamfer.mean()

        loss_normal=np.append(loss_normal,loss_normal_m)
        loss_chamfer=np.append(loss_chamfer,loss_chamfer_m)
        print(loss_normal,'****',loss_chamfer)

        mean_data.insert(0, [loss_chamfer_m, loss_normal_m])


        csv_file = os.path.join( os.path.join( stats_parent, 'stats_3d'),'3d_data.csv')
        print(csv_file)

        data = zip(loss_normal,loss_chamfer )

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['loss_normal','loss_chamfer'])
            writer.writerows(data)
        
        # Save a CSV containing only the mean values (2D means if available and 3D means)

        means_csv = os.path.join(stats_parent, 'ours.csv')


        mean_fields = ["name", 'CD', 'NC','ssim_normals', 'lpips_normals', 'psnr_normals',
                    'ssim_rgb', 'lpips_rgb', 'psnr_rgb', 'IOU']


        with open(means_csv, 'w', newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(mean_fields)
            # Flatten mean_data into a single row matching mean_fields and write as one CSV row
            flat = [round(val, 3) for row in mean_data for val in row]
            # remove the 5th element (index 4) if it exists
            mean_fields.pop(5)
            flat.pop(5)

            # Ensure the flattened row matches header length (pad or trim if necessary)
            if len(flat) < len(mean_fields):
                flat += [''] * (len(mean_fields) - len(flat))
            elif len(flat) > len(mean_fields):
                flat = flat[:len(mean_fields)]
            flat.insert(0, "ours")
            moyenne.insert(0, "mean")
            std.insert(0, "std")
            writer.writerow(flat)
            writer.writerow(moyenne)
            writer.writerow(std)
    else:

        moyenne = [0.870,	0.125,	20.827,	0.900,	0.073,	23.229,	0.976]
        std = [0.00030,	0.00053,	0.02335,	0.00028,	0.00056,	0.06945,	0.00018]
        print("3D evaluation skipped due to missing target mesh data.")
        # Save a CSV containing only the mean values (2D means only)
        means_csv = os.path.join(stats_parent, 'ours.csv')
        # mean_fields = ['ssim_normals', 'lpips_normals', 'psnr_normals', 'IOU_normals',
        #             'ssim_rgb', 'lpips_rgb', 'psnr_rgb', 'IOU']
        # Prepare the header and data row so they match in length
        header = ['ssim_normals', 'lpips_normals', 'psnr_normals',
                  'ssim_rgb', 'lpips_rgb', 'psnr_rgb', 'IOU']
        flat = [round(val, 3) for row in mean_data for val in row]

        # mean_fields.pop(3)
        flat.pop(3)
        # Remove the 4th element (index 3) from both header and flat if flat is longer than header
        # if len(flat) > len(header):
        #     flat.pop(3)
        # Pad or trim flat to match header length
        if len(flat) < len(header):
            flat += [''] * (len(header) - len(flat))
        elif len(flat) > len(header):
            flat = flat[:len(header)]
        
        flat.insert(0, "ours")
        moyenne.insert(0, "mean")
        std.insert(0, "std")
        header.insert(0, "name")
        with open(means_csv, 'w', newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(header)
            writer.writerow(flat)
            writer.writerow(moyenne)
            writer.writerow(std)

    
