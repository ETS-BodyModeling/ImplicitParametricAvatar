import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.io import load_obj
from pytorch3d.ops import SubdivideMeshes


import torch

import torch

def subdivide_lbs_weights(lbs_weights, edges, batch_size=1):
    """
    Compute new LBS weights for a subdivided mesh in a memory-efficient way.

    Args:
        lbs_weights (torch.Tensor): (V, B) tensor of LBS weights.
        edges (torch.Tensor): (E, 2) tensor containing edge vertex indices.
        batch_size (int, optional): Number of edges to process at a time (reduces memory load).
    
    Returns:
        new_lbs_weights (torch.Tensor): (V + E, B) tensor with new vertex weights.
    """
    V, B = lbs_weights.shape  # Original vertices, number of bones
    E = edges.shape[0]        # Number of edges

    new_weights = torch.empty((E, B), dtype=lbs_weights.dtype, device=lbs_weights.device)

    # Process edges in batches to avoid OOM error
    for i in range(0, E, batch_size):
        batch_edges = edges[i:i+batch_size]
        new_weights[i:i+batch_size] = lbs_weights[batch_edges].mean(dim=1)

    # Concatenate original and new weights
    new_lbs_weights = torch.cat([lbs_weights, new_weights], dim=0)  # (V + E, B)

    return new_lbs_weights


def interpolate_weights(W, original_mesh, subdivided_mesh):
    """
    Interpolates LBS weights for the new vertices after mesh subdivision.

    Parameters:
    - W: torch.Tensor of shape (N, V, J+1)
      The original LBS weights for each vertex.
    - original_mesh: pytorch3d.structures.Meshes
      The original low-resolution mesh.
    - subdivided_mesh: pytorch3d.structures.Meshes
      The high-resolution subdivided mesh.

    Returns:
    - W_new: torch.Tensor of shape (N, V_subdiv, J+1)
      The new LBS weights compatible with the subdivided mesh.
    """

    # Extract information about original and subdivided meshes
    V_orig = original_mesh.verts_packed().shape[0]  # Number of original vertices
    V_subdiv = subdivided_mesh.verts_packed().shape[0]  # Number of vertices after subdivision
    F_orig = original_mesh.faces_packed()  # Original faces (F, 3)

    # Original vertex weights
    W_orig = W[:, :V_orig, :]  # (N, V, J+1)

    # Prepare space for new weights
    W_new = torch.zeros((W.shape[0], V_subdiv, W.shape[2]), device=W.device)  # (N, V_subdiv, J+1)
    W_new[:, :V_orig, :] = W_orig  # Copy original weights for original vertices

    # Interpolate weights for new vertices created during subdivision
    # SubdivideMeshes creates new vertices for edges in the original mesh
    edges_to_faces = {}  # Maps edges to their respective faces
    for face_idx, face in enumerate(F_orig):
        v0, v1, v2 = face.tolist()
        edges = [tuple(sorted((v0, v1))), tuple(sorted((v1, v2))), tuple(sorted((v2, v0)))]
        for edge in edges:
            if edge not in edges_to_faces:
                edges_to_faces[edge] = []
            edges_to_faces[edge].append(face_idx)

    # Get the new vertices by comparing original and subdivided mesh vertex counts
    new_vertex_start = V_orig
    for edge, face_idxs in edges_to_faces.items():
        v0, v1 = edge

        # Find the new vertex index for this edge
        if new_vertex_start < V_subdiv:
            W_new[:, new_vertex_start, :] = (W_orig[:, v0, :] + W_orig[:, v1, :]) / 2
            new_vertex_start += 1

    return W_new


# Example usage
verts, faces, aux  = load_obj("/home/fares/ImplicitParametricAvatar/data/smplx_uv_simple.obj")  # Load the original mesh
original_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
edges = original_mesh[0].edges_packed()

# verts_sub, faces_sub, aux_sub = load_obj("/home/fares/ImplicitParametricAvatar/data/smplx_uv_simple_sub.obj")  # Load the subdivided mesh
# subdivided_mesh = Meshes(verts=[verts_sub], faces=[faces_sub.verts_idx])

subdivided_mesh = SubdivideMeshes()(original_mesh)

# Original LBS Weights (Assume it has shape [N, V, J+1])
W = torch.load("/home/fares/ImplicitParametricAvatar/data/W_tensor.pt")

W_new = subdivide_lbs_weights(W.squeeze(), edges)
# Compute new interpolated weights
# W_new = interpolate_weights(W, original_mesh, subdivided_mesh)
torch.save(W_new.squeeze(0), "/home/fares/ImplicitParametricAvatar/data/W_sub_tensor.pt")
print(W_new.shape, W.shape)
# W_new is now compatible with the subdivided mesh!
