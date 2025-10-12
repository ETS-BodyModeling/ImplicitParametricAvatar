import os
from os import walk
import trimesh
import numpy as np
from numpy import linalg as LA  

def rgb(minimum, maximum, value):
    #used for the heatmap render of the faces
    if (value > maximum):
        value = maximum
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


folder_name = "all"
out_path_recons='/home/ext_fares_podform3d_com/test/output_finale/'
out_path_recons= os.path.join(out_path_recons, folder_name)
for (dirpath, dirnames,filenames) in walk(out_path_recons):
  for i,dir in enumerate(dirnames) :
    print(dir)
    path_mesh_t = os.path.join(out_path_recons, dir, 'smpl_final1.obj')
    path_mesh_d = os.path.join(out_path_recons, dir, 'smpl_final_clothes.obj')
    mesh_t = trimesh.load_mesh(path_mesh_t)
    mesh_d = trimesh.load_mesh(path_mesh_d)
    f = open('/home/ext_fares_podform3d_com/test/test/' + dir +  '_heatmap.obj', "w+")
    for ids in range(mesh_d.vertices.shape[0]):
        temp = LA.norm(mesh_t.vertices[ids] - mesh_d.vertices[ids])
        # temp = torch.dist(points_test[ids], target_points[h, :])
        r, g, b = rgb(0, 10, temp * 1000)
        f.write("v {} {} {} {} {} {}\n".format(mesh_d.vertices[ids][0], mesh_d.vertices[ids][1], mesh_d.vertices[ids][2], r, g, b))
    f = open('/home/ext_fares_podform3d_com/test/test/' + dir +  '_norm.txt', "w+")
    f.write("norm D = {}\n".format(LA.norm(mesh_t.vertices - mesh_d.vertices)))