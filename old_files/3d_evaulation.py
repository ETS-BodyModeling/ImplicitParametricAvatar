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
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from trimesh.exchange.load import Trimesh
import numpy as np
import csv
from pytorch3d.structures import Meshes
import scipy.spatial.distance as distance

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


path_in='/home/ext_fares_podform3d_com/test/data/target_mesh'
path_pred='/home/ext_fares_podform3d_com/test/output/base_1'
if not os.path.exists(os.path.join( path_pred, 'stats')):
    # Create the directory if it does not exist
    os.mkdir(os.path.join( path_pred, 'stats'))
loss_chamfer=[]
loss_normal=[]
loss_iou3d=[]

for (_, dirnames, _) in os.walk(path_in):
  for j,dir in enumerate(dirnames):
    print(j)
    # if i==2:
    #   break
    path_mesh_t=os.path.join(path_in, dir +'/frontmesh-f00001.obj')
    print(path_mesh_t)
    with open(os.path.join(path_in, dir +'/gender.txt'), "r") as file:
      gender = file.read().replace(" ", "").replace("\n", "")
    print(gender)


    mesh_t = trimesh.load_mesh(path_mesh_t)
    #mesh_t.export('/content/xavatar.obj')
    mesh_t_torch=torch.tensor(mesh_t.vertices, dtype=torch.float32, requires_grad=True, device=device)


    mesh_f = trimesh.load_mesh(path_pred+'/result_'+gender+'-'+dir+'_512/smpl_final_clothes.obj')
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


csv_file = os.path.join( os.path.join( path_pred, 'stats'),'3d_data.csv')
print(csv_file)

data = zip(loss_normal,loss_chamfer )

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['loss_normal','loss_chamfer'])
    writer.writerows(data)
