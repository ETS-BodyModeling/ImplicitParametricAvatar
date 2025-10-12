from pytorch3d.io import load_obj, save_obj
import torch
import numpy as np
import trimesh
from collections import defaultdict

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
    8164, 8175, 18698, 18647, 18624, 8101, 8176, 18699


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
    [8488, 8500, 8544]


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
    # elif i in [20]:
    #     faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
    #                                                                      val2[0],
    #                                                                      val3[0]]])),
    #                                   dim=0)
    elif i in [5, 7, 8, 34, 36]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[1],
                                                                         val2[0],
                                                                         val3[0]]])),
                                      dim=0)
    elif i in [4, 33, 51]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[0],
                                                                         val3[1]]])),
                                      dim=0)
    elif i in [6, 22]:
        faces_uvs_new_add = torch.cat((faces_uvs_new_add, torch.tensor([[val1[0],
                                                                         val2[1],
                                                                         val3[0]]])),
                                      dim=0)

    elif i in [35]:
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
