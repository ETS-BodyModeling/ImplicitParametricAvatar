from pytorch3d.io import load_obj, save_obj
import torch
import numpy as np
import trimesh
from collections import defaultdict

def merge_uv_points_with_mean(verts_uvs, faces_uvs, index_a, index_b):
    """
    Merge two UV points in a mesh by replacing them with their mean
    without altering the 3D geometry of the mesh.
    Parameters:
    - verts_uvs: torch.Tensor of shape (1, V, 2), UV coordinates
    - faces_uvs: torch.Tensor of shape (1, F, 3), texture indices
    - index_a: int, index of the first UV point
    - index_b: int, index of the second UV point

    Returns:
    - updated_verts_uvs: torch.Tensor, modified UV coordinates
    - updated_faces_uvs: torch.Tensor, updated texture indices
    """
    # Calculate the mean of the two UV points
    mean_uv = (verts_uvs[0, index_a] + verts_uvs[0, index_b]) / 2.0

    # verts_a = torch.cat((verts_uvs[0, index_a], torch.zeros(1)), 0)
    # verts_b = torch.cat((verts_uvs[0, index_b], torch.zeros(1)), 0)

    # trimesh.Trimesh(vertices=verts_a.numpy(), faces=None).export("data/index_a.obj")
    # trimesh.Trimesh(vertices=verts_b.numpy(), faces=None).export("data/index_b.obj")

    # Replace the two points with their mean
    verts_uvs[0, index_a] = mean_uv
    verts_uvs[0, index_b] = mean_uv

    # Update face indices to point to a single UV index (index_a)
    faces_uvs[faces_uvs == index_b] = index_a

    return verts_uvs, faces_uvs


def save_obj_with_uv_tensors(file_path, vertices, uv_coords, faces, faces_uv):
    """
    Save a 3D model in OBJ format with UV coordinates from tensors.

    Args:
        file_path (str): Path to save the OBJ file.
        vertices (torch.Tensor): Tensor of shape (n, 3) for vertex coordinates.
        uv_coords (torch.Tensor): Tensor of shape (m, 2) for UV texture coordinates.
        faces (torch.Tensor): Tensor of shape (k, 3) for vertex indices of faces.
        faces_uv (torch.Tensor): Tensor of shape (k, 3) for UV indices of faces.
    """
    with open(file_path, 'w') as obj_file:
        # Write vertices
        for v in vertices:
            x, y, z = v.tolist()
            obj_file.write(f"v {x} {y} {z}\n")
        
        # Write UV coordinates
        for uv in uv_coords:
            u, v = uv.tolist()
            obj_file.write(f"vt {u} {v}\n")
        
        # Write faces, combining vertex indices and UV indices
        for face, uv_face in zip(faces, faces_uv):
            face_str = " ".join([f"{v+1}/{uv+1}" for v, uv in zip(face.tolist(), uv_face.tolist())])
            obj_file.write(f"f {face_str}\n")


list_faces_remove = torch.tensor([
    18678, 8172, 18695, 8191, 18714, 18709, 8186, 18633, 8110, 18665, 8142, 8185, 18708, 8169, 18692,
    8143, 18666, 8192, 8116, 18677, 8154, 18639, 18693, 8155, 18715,
    # second toe
    8147, 18670, 8170, 18693, 8187, 18710, 8146, 18669, 8112, 18635, 8188, 18711, 18664, 8141, 18691,
    8168, 8183, 18706, 8140, 18663, 8109, 18632, 8184, 18707,
    # third toe
    18660, 8137, 8167, 18690, 8181, 18704, 8136, 18659, 8107, 18630, 8182, 18705, 18650, 8127, 8165, 
    18688, 18700, 8177, 8126, 18649, 18625, 8102, 8178, 18701,
    # big toe
    18654, 8131, 18689, 8166, 8179, 18702, 8130, 18653, 18627, 8104, 8180, 18703, 18648, 8125, 18687,
    8164, 8175, 18698, 18647, 18624, 8101, 8176, 18699,
    # left feet
    14853, 4322, 14789, 4258, 14938, 5321, 15850, 14848, 4317, 5226, 4415, 15755, 5237, 15766, 4323, 14854, 4399, 14930, 14946, 15851, 5322, 5130, 15659,
    14853, 4321, 15847, 5318, 14942, 4411, 4398, 14929, 4332, 14863, 14814, 4283, 14852,
    4306, 14837, 14861, 4330, 4613, 14144, 16223, 5694, 15915, 5386, 15144, 14839, 14871, 4340,
    14875, 4344, 4400, 14931, 4325, 14856, 4343, 14874, 14849, 4318, 14838, 4307, 14839, 4308,
    14869, 4338, 14862, 4331, 16165, 5636, 14791, 4260, 14864, 4333, 14876, 4345, 8747, 19269,
    5128, 15657, 4328, 14859, 14922, 4391, 15792, 5263, 4334, 15889, 5360, 14878,  4347, 14868, 4337, 14944, 4413, 4371


                                  ])



verts, faces, aux = load_obj("data/smplx_uv.obj")
verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)

masque = ~torch.isin(torch.arange(faces.verts_idx.size(0)), list_faces_remove)
# Tenseur résultant
faces_new = faces.verts_idx[masque].squeeze()
faces_uvs_new = faces_uvs[:, masque].squeeze()


# list_faces_add = torch.tensor([
#     [5846, 5785, 5802]
#                                   ])

faces_temp = faces.verts_idx.reshape(-1)
faces_uvs_temp = faces_uvs.reshape(-1)
# Create a defaultdict with sets as the default value type
my_dict = defaultdict(set)
# Add elements to the sets
for i in range(faces_temp.shape[0]):
    my_dict[faces_temp[i].item()].add(faces_uvs_temp[i].item())

# Row to add
new_rows = torch.tensor([
    # small toe
    [8561, 8531, 8516],
    [8516, 8531, 8532],
    [8532, 8565, 8516],
    [8516, 8565, 8515],
    [8515, 8565, 8473],
    [8473, 8557, 8515],
    [8557, 8473, 8478],
    [8478, 8473, 8472],
    [8478, 8472, 8475],
    [8475, 8472, 8525],
    [8525, 8472, 8526],
    [8526, 8472, 8536],
    [8536, 8533, 8526],
    [8526, 8533, 8559],
    # second toe
    [8552, 8519, 8505],
    [8505, 8519, 8520],
    [8505, 8520, 8560],
    [8506, 8505, 8560],
    [8506, 8560, 8460],
    [8460, 8560, 8477],
    [8460, 8477, 8461],
    [8461, 8477, 8554],
    [8554, 8477, 8476],
    [8476, 8511, 8554],
    [8511, 8476, 8512],
    [8476, 8524, 8512],
    [8512, 8524, 8521],
    [8521, 8555, 8512],
    # 3 toe
    [8551, 8504, 8494],
    [8494, 8504, 8493],
    [8493, 8504, 8503],
    [8503, 8459, 8493],
    [8459, 8547, 8493],
    [8547, 8459, 8462],
    [8462, 8470, 8547],
    [8470, 8462, 8467],
    [8467, 8462, 8502],
    [8502, 8462, 8553],
    [8553, 8501, 8502],
    [8501, 8553, 8510],
    [8501, 8510, 8507],
    [8501, 8507, 8549],
    # big toe
    [8540, 8496, 8479],
    [8479, 8496, 8480],
    [8480, 8496, 8495],
    [8495, 8550, 8480],
    [8550, 8465, 8480],
    [8465, 8550, 8469],
    [8469, 8543, 8465],
    [8543, 8469, 8468],
    [8543, 8468, 8499],
    [8499, 8487, 8543],
    [8487, 8499, 8500],
    [8500, 8488, 8487],
    [8488, 8500, 8544],

# left feet
    # big toe
    [5846, 5785, 5802],
    [5802, 5785, 5786],
    [5786, 5801, 5802],
    [5786, 5771, 5801],
    [5771, 5856, 5801],
    [5771, 5774, 5775],
    [5771, 5849, 5805],
    [5771, 5775, 5856],
    [5771, 5805, 5774],
    [5805, 5849, 5793],
    [5805, 5793, 5806],
    [5806, 5793, 5794],
    [5806, 5794, 5850],
#     second toe
    [5857, 5800, 5810],
    [5810, 5800, 5809],
    [5809, 5800, 5799],
    [5809, 5799, 5765],
    [5765, 5799, 5768],
    [5768, 5799, 5853],
    [5853, 5776, 5768],
    [5776, 5859, 5768],
    [5859, 5776, 5816],
    [5816, 5776, 5773],
    [5773, 5808, 5816],
    [5808, 5807, 5816],
    [5816, 5807, 5813],
    [5813, 5807, 5855],
#     third toe
    [5858, 5811, 5825],
    [5825, 5811, 5826],
    [5826, 5811, 5812],
    [5812, 5866, 5826],
    [5866, 5812, 5783],
    [5783, 5812, 5766],
    [5766, 5767, 5783],
    [5783, 5767, 5860],
    [5783, 5860, 5817],
    [5783, 5817, 5782],
    [5782, 5817, 5830],
    [5830, 5817, 5827],
    [5827, 5817, 5818],
    [5818, 5861, 5827],
#     small toe
    [5867, 5822, 5837],
    [5837, 5822, 5838],
    [5838, 5822, 5871],
    [5871, 5822, 5779],
    [5779, 5822, 5821],
    [5779, 5821, 5863],
    [5779, 5863, 5784],
    [5784, 5781, 5779],
    [5779, 5781, 5778],
    [5778, 5781, 5831],
    [5831, 5832, 5778],
    [5832, 5842, 5778],
    [5842, 5832, 5839],
    [5839, 5832, 5865 ]

])
faces_new_add = torch.cat((faces_new, new_rows), dim=0)

faces_uvs_new_add = faces_uvs_new.clone()
for i in range (new_rows.shape[0]):
    val1=list(my_dict[new_rows[i, 0].item()])
    val2=list(my_dict[new_rows[i, 1].item()])
    val3=list(my_dict[new_rows[i, 2].item()])
    print(i, val1, val2, val3)

    if i in [21]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[1],
                                                                         val2[1],
                                                                         val3[0]]])),
                                      dim=0)
    elif i in [19 + 55, 33 + 55]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[1],
                                                                         val3[1]]])),
                                      dim=0)
    elif i in [5, 7, 8, 34, 36,  6 + 55, 8 + 55, 18 + 55, 32 + 55, 49 + 55]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[1],
                                                                         val2[0],
                                                                         val3[0]]])),
                                      dim=0)
    elif i in [4, 33, 51, 17 + 55, 31 + 55]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[0],
                                                                         val3[1]]])),
                                      dim=0)
    elif i in [6, 22]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[1],
                                                                         val3[0]]])),
                                      dim=0)

    elif i in [35, 5 + 55, 48 + 55]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[1],
                                                                         val2[0],
                                                                         val3[1]]])),
                                      dim=0)
    else:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                            val2[0],
                                                                            val3[0]]])),
                                        dim=0)


verts_uvs_3d = torch.cat((verts_uvs.squeeze(), torch.zeros((verts_uvs.shape[1], 1))), 1)

# mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d.numpy(), faces=faces_uvs.numpy().squeeze())
# mesh_uv.export("data/smplx_uv_3d.obj")

mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d.numpy(), faces=faces_uvs_new.numpy().squeeze())
mesh_uv.export("data/smplx_uv_3d_remove.obj")

mesh_remove = trimesh.Trimesh(vertices=verts.numpy(), faces=faces_new.numpy().squeeze())
mesh_remove.export("data/smplx_remove.obj")

mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d.numpy(), faces=faces_uvs_new_add.numpy().squeeze())
mesh_uv.export("data/smplx_uv_3d_add.obj")

mesh_add = trimesh.Trimesh(vertices=verts.numpy(), faces=faces_new_add.numpy().squeeze())
mesh_add.export("data/smplx_add.obj")


# output_path= "data/smplx_remove.obj"
# save_obj(
#     output_path,
#     verts,
#     faces_new,
#     verts_uvs=aux.verts_uvs if aux else None,
#     faces_uvs=faces_uvs_new.squeeze() if aux and faces else None,
# )

# print(aux.verts_uvs.shape, faces_uvs_new_add.shape)
# output_path= "data/smplx_uv_simple.obj"
# save_obj(
#     output_path,
#     verts,
#     faces_new_add,
#     verts_uvs=aux.verts_uvs if aux else None,
#     faces_uvs=faces_uvs_new_add.squeeze() if aux and faces else None,
# )

save_obj_with_uv_tensors("data/smplx_uv_simple.obj", verts.squeeze(), aux.verts_uvs.squeeze(), faces_new_add.squeeze(), faces_uvs_new_add.squeeze())


# verts, faces, aux = load_obj("data/smplx_uv_simple.obj")
# Merge two UV points (e.g., point 0 and point 1)
index_a = 11281  # Index of the first UV point
index_b = 9508  # Index of the second UV point
verts_uvs = aux.verts_uvs.unsqueeze(0)  # (1, V, 2)
faces_uvs = faces_uvs_new_add

# 
# print(faces_uvs_new_add[20767], faces_uvs_new_add[20768], faces_uvs_new_add[8090])
# # print(faces_uvs_new_add[20770], faces_uvs_new_add[8068], faces_uvs_new_add[18518])
# print(faces_uvs_new_add[20769], faces_uvs_new_add[20770], faces_uvs_new_add[20771])
# exit()

# Merge the UV points with their mean
updated_verts_uvs, updated_faces_uvs = merge_uv_points_with_mean(verts_uvs, faces_uvs, index_a, index_b)

index_a = 9565  # Index of the first UV point
index_b = 9517  # Index of the second UV point

updated_verts_uvs, updated_faces_uvs = merge_uv_points_with_mean(verts_uvs, faces_uvs, index_a, index_b)

save_obj_with_uv_tensors("data/smplx_uv_simple.obj", verts.squeeze(), updated_verts_uvs.squeeze(), faces_new_add.squeeze(), updated_faces_uvs.squeeze())

verts_uvs_3d_merged = torch.cat((updated_verts_uvs.squeeze(), torch.zeros((verts_uvs.shape[1], 1))), 1)
mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d_merged.numpy(), faces=updated_faces_uvs.numpy().squeeze())
mesh_uv.export("data/smplx_uv_3d_merged.obj")
