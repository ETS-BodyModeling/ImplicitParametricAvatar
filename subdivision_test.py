import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.io import load_obj, save_obj

import os
import argparse
from pathlib import Path
import sys
import trimesh
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

def subdivide_mesh_with_uv(mesh: Meshes, mesh_uv: Meshes):
    """
    Subdivide a PyTorch3D mesh and its corresponding UV map.

    Parameters
    ----------
    mesh : pytorch3d.structures.Meshes
        The input mesh to subdivide.
    verts_uvs : torch.Tensor
        UV coordinates for the original mesh vertices. Shape: (V, 2).
    faces_uvs : torch.Tensor
        UV face indices for the original mesh. Shape: (F, 3).

    Returns
    -------
    subdivided_mesh : Meshes
        The subdivided PyTorch3D mesh.
    subdivided_verts_uvs : torch.Tensor
        UV coordinates for the subdivided mesh. Shape: (V_new, 2).
    subdivided_faces_uvs : torch.Tensor
        UV face indices for the subdivided mesh. Shape: (F_new, 3).
    """
    # Step 1: Subdivide the 3D geometry
    subdivider = SubdivideMeshes()
    subdivided_mesh = subdivider(mesh)
    
    
    # Step 2: Subdivide the UV map

    subdivided_mesh_uv = subdivider(mesh_uv)


    return subdivided_mesh, subdivided_mesh_uv

_ROOT_DIR = Path(__file__).resolve()
sys.path.append(str(_ROOT_DIR))
root_path = _ROOT_DIR.parent
# mesh=load_obj(os.path.join(root_path, "data", 'smplx_uv_simple.obj'))
verts, faces, aux = load_obj(os.path.join(root_path, "data", 'smplx_uv_simple.obj'))
verts_uvs = aux.verts_uvs[None, ...].squeeze()  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...].squeeze()  # (1, F, 3)
mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
mesh_uv = Meshes(verts=[(torch.cat((verts_uvs, torch.zeros((verts_uvs.shape[0], 1))), 1))], faces=[faces_uvs])

subdivided_mesh, subdivided_mesh_uv = subdivide_mesh_with_uv(mesh, mesh_uv)
save_obj_with_uv_tensors("data/smplx_uv_simple_sub.obj", subdivided_mesh.verts_packed(), subdivided_mesh_uv.verts_packed()[:, :2], subdivided_mesh.faces_packed(), subdivided_mesh_uv.faces_packed())

# verts_uvs_3d_merged = torch.cat((subdivided_mesh_uv.verts_packed(), torch.zeros((subdivided_mesh_uv.verts_packed().shape[0], 1))), 1)
mesh_uv = trimesh.Trimesh(vertices=subdivided_mesh_uv.verts_packed().detach().cpu().numpy(), faces=subdivided_mesh_uv.faces_packed().detach().cpu().numpy())
mesh_uv.export("data/smplx_uv_3d_sub.obj")
