from pytorch3d.io import load_obj, save_obj
import torch
import numpy as np
import trimesh
from collections import defaultdict

import torch

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

# list left feet
list_faces_remove = torch.tensor([14853, 4322, 14789, 4258, 14938, 5321, 15850, 14848, 4317, 5226, 4415, 15755, 5237, 15766, 4323, 14854, 4399, 14930, 14946, 15851, 5322, 5130, 15659,
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
print(faces_temp[np.where(faces_uvs_temp == 5812)])
exit()
# Create a defaultdict with sets as the default value type
my_dict = defaultdict(set)
# Add elements to the sets
for i in range(faces_temp.shape[0]):
    my_dict[faces_temp[i].item()].add(faces_uvs_temp[i].item())

# Row to add
new_rows = torch.tensor([
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

    if i in [5, 48]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[1],
                                                                         val2[0],
                                                                         val3[1]]])),
                                      dim=0)
    elif i in [6, 8, 18, 32, 49]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[1],
                                                                         val2[0],
                                                                         val3[0]]])),
                                      dim=0)
    elif i in [17, 48]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[0],
                                                                         val3[1]]])),
                                      dim=0)
    elif i in [19, 33]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[1],
                                                                         val3[1]]])),
                                      dim=0)

    elif i in [31]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[0],
                                                                         val3[1]]])),
                                      dim=0)
    else:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[0],
                                                                         val3[0]]])),
                                      dim=0)


verts_uvs_3d = torch.cat((verts_uvs.squeeze(), torch.zeros((verts_uvs.shape[1], 1))), 1)

mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d.numpy(), faces=faces_uvs.numpy().squeeze())
mesh_uv.export("data/smplx_uv_3d.obj")

mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d.numpy(), faces=faces_uvs_new.numpy().squeeze())
mesh_uv.export("data/smplx_uv_3d_remove.obj")

mesh_uv = trimesh.Trimesh(vertices=verts_uvs_3d.numpy(), faces=faces_uvs_new_add.numpy().squeeze())
mesh_uv.export("data/smplx_uv_3d_add.obj")


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

# mesh_simple = trimesh.load_mesh('/home/fares/ImplicitParametricAvatar/data/smplx_uv_simple.obj', process=False, maintain_order=True)

# print(mesh_simple.faces.reshape(-1).max())
# exit()  