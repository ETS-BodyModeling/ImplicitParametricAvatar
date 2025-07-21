from pytorch3d.ops.knn import knn_gather, knn_points
import torch
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageOps
from pytorch3d.io import load_obj, save_obj
import trimesh
import numpy as np
import os
import logging
# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filename="app.log",  # Log file name
    filemode="w"  # File mode: 'w' to overwrite, 'a' to append
)


def loss_with_constraint(variable, a, b, lambda_weight):
    penalty = 0

    # penalty for values less than 'a'
    penalty += torch.relu(a - variable).sum()

    # penalty for values greater than 'b'
    penalty += torch.relu(variable - b).sum()
    return lambda_weight * penalty


def init_params(model, device):
    betas = torch.zeros((1, model.num_betas), dtype=torch.float32, device=device, requires_grad=True)
    expression = torch.zeros(
        [1, model.num_expression_coeffs], dtype=torch.float32, device=device, requires_grad=True)
    body_pose = torch.zeros(
        (1, 21 * 3), dtype=torch.float32)
    body_pose[0, 3 * 15 + 2] = -1
    body_pose[0, 3 * 16 + 2] = 1
    # body_pose= torch.tensor(body_pose, dtype=torch.float32, device=device, requires_grad=True)
    body_pose = body_pose.clone().detach().to(device).requires_grad_(True)
    # body_pose.requires_grad_=True

    global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device, requires_grad=True)
    # transl=torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)

    s1_params = torch.ones(1, requires_grad=True).to(device)
    rx = torch.zeros((1,), dtype=torch.float32, requires_grad=True).to(device)
    ry = torch.zeros((1,), dtype=torch.float32, requires_grad=True).to(device)
    rz = torch.zeros((1,), dtype=torch.float32, requires_grad=True).to(device)
    D = torch.zeros((10475, 3), dtype=torch.float32, device=device, requires_grad=True)
    left_hand_pose = torch.zeros((1, 15 * 3), device=device, dtype=torch.float32)
    for i in range(15):
        left_hand_pose[0, 3 * i + 2] = 0.5
    left_hand_pose[0, 38] = -0.0
    left_hand_pose[0, 37] = -0.0
    left_hand_pose[0, 36] = -0.0

    right_hand_pose = -left_hand_pose

    # right_hand_pose= torch.tensor(right_hand_pose.detach(), dtype=torch.float32, device=device, requires_grad=True)
    right_hand_pose = right_hand_pose.clone().detach().to(device).requires_grad_(True)
    left_hand_pose = left_hand_pose.clone().detach().to(device).requires_grad_(True)
    # left_hand_pose= torch.tensor(left_hand_pose.detach(), dtype=torch.float32, device=device, requires_grad=True)
    # left_hand_pose.requires_grad=True
    # right_hand_pose.requires_grad=True
    return betas, body_pose, expression, global_orient, s1_params, rx, ry, rz, D, left_hand_pose.to(
        device), right_hand_pose.to(device)


def extract_texture_pifu(pifu, smpl_clothed, path_uv, path_save, uv_size=512, name ="texture.png"):
    _, faces, aux = load_obj(path_uv)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    verts_uvs_3d = torch.cat((verts_uvs.squeeze(), torch.zeros((verts_uvs.shape[1], 1))), 1)


    mesh_uv = trimesh.Trimesh(vertices=np.asarray(verts_uvs_3d), faces=np.asarray(faces_uvs.squeeze()), process=False, maintain_order=True)
    # mesh_uv.export('output/uv.obj')

    logging.info('smpl_clothed.vertices.shape: %s', smpl_clothed.vertices.shape)

    uv_image = torch.zeros((uv_size, uv_size, 3))
    x = np.arange(0, 1, 1 / uv_size) + 1 / (2 * uv_size)
    xv, yv = np.meshgrid(x, x, indexing='xy')
    points = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1), np.zeros((uv_size ** 2, 1))), axis=1)

    # mesh_point = trimesh.Trimesh(vertices=points,faces=None)
    # mesh_point.export('mesh_point.obj')

    res = trimesh.proximity.ProximityQuery(mesh_uv)
    closest_tringles = res.on_surface(points)

    bary_coor = trimesh.triangles.points_to_barycentric(verts_uvs_3d[faces_uvs[0, closest_tringles[2]]], points,
                                                        method='cross')
    faces_matched = faces[0][closest_tringles[2]]

    vertices_smpl = smpl_clothed.vertices

    V0 = np.concatenate((np.multiply(vertices_smpl[faces_matched[:, 0]][:, 0], bary_coor[:, 0]).reshape(-1, 1),
                         np.multiply(vertices_smpl[faces_matched[:, 0]][:, 1], bary_coor[:, 0]).reshape(-1, 1),
                         np.multiply(vertices_smpl[faces_matched[:, 0]][:, 2], bary_coor[:, 0]).reshape(-1, 1)), axis=1)
    V1 = np.concatenate((np.multiply(vertices_smpl[faces_matched[:, 1]][:, 0], bary_coor[:, 1]).reshape(-1, 1),
                         np.multiply(vertices_smpl[faces_matched[:, 1]][:, 1], bary_coor[:, 1]).reshape(-1, 1),
                         np.multiply(vertices_smpl[faces_matched[:, 1]][:, 2], bary_coor[:, 1]).reshape(-1, 1)), axis=1)
    V2 = np.concatenate((np.multiply(vertices_smpl[faces_matched[:, 2]][:, 0], bary_coor[:, 2]).reshape(-1, 1),
                         np.multiply(vertices_smpl[faces_matched[:, 2]][:, 1], bary_coor[:, 2]).reshape(-1, 1),
                         np.multiply(vertices_smpl[faces_matched[:, 2]][:, 2], bary_coor[:, 2]).reshape(-1, 1)), axis=1)
    V = V0 + V1 + V2
    # visualisation barycentrique mesh
    #   mesh_bary = trimesh.Trimesh(vertices=V,faces=None)
    #   mesh_bary.export('mesh_bary.obj')

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pifu.vertices)
    _, indices = neigh.kneighbors(V, return_distance=True)

    p_colors = pifu.visual.vertex_colors[indices].squeeze()
    print(p_colors.shape, p_colors.max(), p_colors.min())
    # print(p_colors.shape, p_colors.max(), p_colors.min())
    im = Image.fromarray(p_colors[:, :3].reshape((uv_size, uv_size, 3), order='C'))
    im_flip = ImageOps.flip(im)
    im_flip.save(os.path.join(path_save, name))


def centroid_mesh(x):
    # x = torch.tensor(mesh)
    return torch.stack([((x[:, 0].max() + x[:, 0].min()) / 2), ((x[:, 1].max() + x[:, 1].min()) / 2),
                        ((x[:, 2].max() + x[:, 2].min()) / 2)])


def recalage_centroide(mesh1, mesh2):
    x = centroid_mesh(mesh1)
    y = centroid_mesh(mesh2)
    return y - x
    # torch.tensor(y)- torch.tensor(x))


def d_point2plan(A, plan):
    return torch.abs(A[0] * plan[0] + A[1] * plan[1] + A[2] * plan[2] + plan[3]) / torch.sqrt_(
        plan[0] ** 2 + plan[1] ** 2 + plan[2] ** 2)


def knn_loss1(src_mesh, PC):
    pk12 = knn_points(src_mesh.verts_packed()[:, :3].unsqueeze(0).float(),
                      PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      src_mesh.verts_packed()[:, :3].unsqueeze(0).float(), K=1)
    # pk21 = knn_points(pc[:, :, :3], mid_points[:, :3].unsqueeze(0), K=3).idx[0]
    Ngb = PC.points_packed()[pk12.idx[0]].squeeze()

    Ngb21 = src_mesh.verts_packed()[pk21.idx[0]].squeeze()
    V21 = PC.points_packed()

    V = src_mesh.verts_packed()
    normals = src_mesh.verts_normals_packed()

    # res=torch.mul(normals,Ngb-V)
    # ress=torch.sum(res,dim=1)
    # idx=torch.where(ress<0)

    d = -normals[:, 0] * V[:, 0] - normals[:, 1] * V[:, 1] - normals[:, 2] * V[:, 2]
    dist = torch.abs(
        Ngb[:, 0] * normals[:, 0] + Ngb[:, 1] * normals[:, 1] + Ngb[:, 2] * normals[:, 2] + d) / torch.sqrt_(
        normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)
    # print('res',res.shape,'ress',ress.shape,'d',d.shape,'dist',dist.shape)
    return torch.sum(dist) + torch.sum(torch.abs(Ngb21 - V21))


def knn_loss2(src_mesh, PC):
    pk12 = knn_points(src_mesh.verts_packed()[:, :3].unsqueeze(0).float(),
                      PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb = PC.points_packed()[pk12.idx[0]].squeeze()

    V = src_mesh.verts_packed()
    normals = src_mesh.verts_normals_packed()

    d = -normals[:, 0] * V[:, 0] - normals[:, 1] * V[:, 1] - normals[:, 2] * V[:, 2]
    dist = torch.abs(
        Ngb[:, 0] * normals[:, 0] + Ngb[:, 1] * normals[:, 1] + Ngb[:, 2] * normals[:, 2] + d) / torch.sqrt_(
        normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)
    return torch.sum(dist)

def knn_loss_sub0(vert, PC):


    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      vert.unsqueeze(0).float(), K=1)
    Ngb21 =vert[pk21.idx[0]].squeeze()
    V21 = PC.points_packed()
    distance = torch.norm(Ngb21 - V21, dim=1)
    # mask = distance <= 0.02
    # distance = distance[mask]
    return torch.sum(distance)

def knn_loss_sub(vert, PC):


    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      vert.unsqueeze(0).float(), K=1)
    Ngb21 =vert[pk21.idx[0]].squeeze()
    V21 = PC.points_packed()
    distance = torch.norm(Ngb21 - V21, dim=1)
    mask = distance <= 0.02
    distance = distance[mask]
    return torch.sum(distance)

def knn_loss_sub1(vert, PC):

    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      vert.unsqueeze(0).float(), K=1)
    Ngb21 =vert[pk21.idx[0]].squeeze()

    pk12 = knn_points(vert.unsqueeze(0).float(), PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb12 = PC.points_packed()[pk12.idx[0]].squeeze()

    V21 = PC.points_packed()
    distance = torch.norm(Ngb21 - V21, dim=1)
    # mask = distance <= 0.01
    mask = distance <= 0.02
    distance = distance[mask]

    V12 = vert.unsqueeze(0).float()
    distance1 = torch.norm(Ngb12 - V12, dim=1)

    return torch.sum(distance) + 10 * torch.sum(distance1)

def knn_loss_sub2(vert, PC):

    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      vert.unsqueeze(0).float(), K=1)
    Ngb21 =vert[pk21.idx[0]].squeeze()

    pk12 = knn_points(vert.unsqueeze(0).float(), PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb12 = PC.points_packed()[pk12.idx[0]].squeeze()

    V21 = PC.points_packed()
    distance = torch.norm(Ngb21 - V21, dim=1)
    # mask = distance <= 0.01
    # distance = distance[mask]

    V12 = vert.unsqueeze(0).float()
    distance1 = torch.norm(Ngb12 - V12, dim=1)

    return torch.sum(distance) + 10 * torch.sum(distance1)

def knn_loss3(src_mesh, PC):
    # pk12 = knn_points(src_mesh.verts_packed()[:, :3].unsqueeze(0).float(),
    #                   PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
    # Ngb = PC.points_packed()[pk12.idx[0]].squeeze()

    # V = src_mesh.verts_packed()
    # normals = src_mesh.verts_normals_packed()

    # d = -normals[:, 0] * V[:, 0] - normals[:, 1] * V[:, 1] - normals[:, 2] * V[:, 2]
    # dist = torch.abs(
    #     Ngb[:, 0] * normals[:, 0] + Ngb[:, 1] * normals[:, 1] + Ngb[:, 2] * normals[:, 2] + d) / torch.sqrt_(
    #     normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)

    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      src_mesh.verts_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb21 = src_mesh.verts_packed()[pk21.idx[0]].squeeze()
    V21 = PC.points_packed()
    distance = torch.norm(Ngb21 - V21, dim=1)
    # mask = distance <= 0.02
    mask = distance <= 0.03
    distance = distance[mask]
    return torch.sum(distance)

def knn_loss4(src_mesh, PC):
    pk12 = knn_points(src_mesh.verts_packed()[:, :3].unsqueeze(0).float(),
                      PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb12 = PC.points_packed()[pk12.idx[0]].squeeze()
    V12 = src_mesh.verts_packed()


    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      src_mesh.verts_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb21 = src_mesh.verts_packed()[pk21.idx[0]].squeeze()
    V21 = PC.points_packed()

    # normals = src_mesh.verts_normals_packed()
    # d = -normals[:, 0] * V[:, 0] - normals[:, 1] * V[:, 1] - normals[:, 2] * V[:, 2]
    # dist = torch.abs(
    #     Ngb[:, 0] * normals[:, 0] + Ngb[:, 1] * normals[:, 1] + Ngb[:, 2] * normals[:, 2] + d) / torch.sqrt_(
    #     normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)
    #  torch.sum(torch.exp(1e1 *torch.norm(Ngb21 - V21, dim=1))) 
    return torch.sum(torch.norm(Ngb21 - V21, dim=1)) + 20 * torch.sum(torch.norm(Ngb12 - V12, dim=1))

# def knn_loss5(src_mesh, PC):
#     pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
#                       src_mesh.verts_packed()[:, :3].unsqueeze(0).float(), K=1)
#     Ngb21 = src_mesh.verts_packed()[pk21.idx[0]].squeeze()
#     V21 = PC.points_packed()
#     return torch.sum(torch.norm(Ngb21 - V21, dim=1))

# def knn_loss4(src_mesh, PC, idx):
#     pk12 = knn_points(src_mesh.verts_packed()[idx][:, :3].unsqueeze(0).float(),
#                       PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
#     Ngb12 = PC.points_packed()[pk12.idx[0]].squeeze()
#     V12 = src_mesh.verts_packed()[idx]


#     pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
#                       src_mesh.verts_packed()[idx][:, :3].unsqueeze(0).float(), K=1)
#     Ngb21 = src_mesh.verts_packed()[idx][pk21.idx[0]].squeeze()
#     V21 = PC.points_packed()

#     # normals = src_mesh.verts_normals_packed()
#     # d = -normals[:, 0] * V[:, 0] - normals[:, 1] * V[:, 1] - normals[:, 2] * V[:, 2]
#     # dist = torch.abs(
#     #     Ngb[:, 0] * normals[:, 0] + Ngb[:, 1] * normals[:, 1] + Ngb[:, 2] * normals[:, 2] + d) / torch.sqrt_(
#     #     normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)
#     #  torch.sum(torch.exp(1e1 *torch.norm(Ngb21 - V21, dim=1))) 
#     return torch.sum(torch.norm(Ngb21 - V21, dim=1)) + 10 * torch.sum(torch.norm(Ngb12 - V12, dim=1))


# def knn_loss3(src_mesh, PC):
#     pk12 = knn_points(src_mesh.verts_packed()[:, :3].unsqueeze(0).float(),
#                       PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)
#     pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
#                       src_mesh.verts_packed()[:, :3].unsqueeze(0).float(), K=1)
#     # pk21 = knn_points(pc[:, :, :3], mid_points[:, :3].unsqueeze(0), K=3).idx[0]
#     Ngb = PC.points_packed()[pk12.idx[0]].squeeze()

#     Ngb21 = src_mesh.verts_packed()[pk21.idx[0]].squeeze()
#     V21 = PC.points_packed()
#     V = src_mesh.verts_packed()
#     normals = src_mesh.verts_normals_packed()
#     return torch.sum(torch.abs(Ngb - V)) + torch.sum(torch.abs(Ngb21 - V21))


def Inside_Penalisation(src_mesh, PC, idx_not_penelized):
    pk12 = knn_points(src_mesh.verts_packed()[:, :3].unsqueeze(0).float(),
                      PC.points_packed()[:, :3].unsqueeze(0).float(), K=1)

    pk21 = knn_points(PC.points_packed()[:, :3].unsqueeze(0).float(),
                      src_mesh.verts_packed()[:, :3].unsqueeze(0).float(), K=1)
    Ngb21 = src_mesh.verts_packed()[pk21.idx[0]].squeeze()
    V21 = PC.points_packed()

    Ngb = PC.points_packed()[pk12.idx[0]].squeeze()
    V = src_mesh.verts_packed()
    normals = src_mesh.verts_normals_packed()

    UN = torch.sum(torch.mul(Ngb - V, normals), axis=1)[idx_not_penelized]

    Ucross = torch.norm(torch.cross(Ngb - V, normals, dim=1), dim=1)[idx_not_penelized]

    loss1 = torch.sum(torch.abs(Ngb - V))

    loss2 = torch.sum(torch.abs(Ngb21 - V21))

    loss3 = torch.mean(torch.exp(-1e2 * UN))
    return loss1, loss2, 100 * loss3