from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
import lpips
import os
# from torchmetrics.functional import ssim, psnr
import torch
from PIL import Image
from torchvision import transforms
# transform = torch.nn.functional.to_tensor
# torchvision.transforms.functional.pil_to_tensor
from rembg import remove
import numpy as np
import cv2
# from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from os import walk
import csv

def calculate_iou(predicted_mask, target_mask):
    # Calculate intersection (logical AND)
    intersection = torch.logical_and(predicted_mask, target_mask).sum().float()

    # Calculate union (logical OR)
    union = torch.logical_or(predicted_mask, target_mask).sum().float()

    return intersection / union

root_path='/home/ext_fares_podform3d_com/test/output_render_base1'
out_path='/home/ext_fares_podform3d_com/test/data/target'

evaluation_pathes=['ours']
names=['ours']
types=['rgb','normals']

transform = transforms.ToTensor()

psnr = PeakSignalNoiseRatio()
lpips_alex = lpips.LPIPS(net='alex')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0) 

i=0
for j,works in enumerate(evaluation_pathes):
  for i,type_select in enumerate(types) :
    print('evaluation de '+type_select+'--->'+ works)
    out_path_target = os.path.join(out_path, type_select)
    out_path_target_silh = os.path.join(out_path, 'silhouette')
    ssim_list=[]
    lpips_list=[]
    psnr_list=[]
    IOU_list=[]
    for (dirpath, dirnames, filenames) in walk(out_path_target):
      for dir in filenames:
        print(dir,'***********')
        im_path_target=os.path.join(out_path_target,dir)
        im_path_target_silh=os.path.join(out_path_target_silh,dir)

        im_path_predicted=os.path.join(os.path.join(os.path.join(root_path,works),type_select),dir)
        im_path_predicted_silh=os.path.join(os.path.join(os.path.join(root_path,works),'silhouette'),dir)

        IOU_path=os.path.join(os.path.join(root_path,works),'IOU')
        if not os.path.exists(IOU_path):
          os.mkdir(IOU_path)


        target_image = Image.open(im_path_target)
        target_image_silh = Image.open(im_path_target_silh)
        # target_image = target_image.resize((256,256))
        # target_image_silh = target_image_silh.resize((256,256))


        target_image_silh_torch = transform(target_image_silh)
        # save_image(target_image_silh_torch, '/content/target_image_silh_torch.png')

        mask_idx = np.where(np.asarray(target_image_silh_torch.squeeze())!=0)



        predicted_image = Image.open(im_path_predicted)
        predicted_image_silh = Image.open(im_path_predicted_silh)



        print(predicted_image.size)
        if predicted_image.size!=(1024,1024) :
          predicted_image = predicted_image.resize((1024,1024))
          predicted_image_silh = predicted_image_silh.resize((1024,1024))

        # predicted_image = predicted_image.resize((256,256))
        # predicted_image_silh = predicted_image_silh.resize((256,256))

        predicted_image_silh_torch = transform(predicted_image_silh)

        mask_idx = np.where(np.asarray(predicted_image_silh_torch.squeeze())!=0)


        IOU_score=calculate_iou(predicted_image_silh_torch,target_image_silh_torch)
        save_image(torch.abs(predicted_image_silh_torch-target_image_silh_torch), IOU_path+'/'+dir)


        target_tensor = transforms.ToTensor()(target_image).unsqueeze(0)/ 255.0
        predicted_tensor =  transforms.ToTensor()(predicted_image).unsqueeze(0)/ 255.0

        save_image(torch.abs(target_tensor-predicted_tensor)*255,  IOU_path+'/'+type_select+dir)

        psnr_score=psnr(predicted_tensor[0,:,mask_idx[0],mask_idx[1]], target_tensor[0,:,mask_idx[0],mask_idx[1]])
        # psnr_score1=psnr(predicted_tensor[0,:,mask_idx[0],mask_idx[1]]*255, target_tensor[0,:,mask_idx[0],mask_idx[1]]*255)

        x, y, w, h = cv2.boundingRect(np.column_stack(mask_idx))

        predicted_tensor = predicted_tensor[:,:,x:x + w,y:y + h]
        target_tensor = target_tensor[:,:,x:x + w,y:y + h]

        # psnr_score1=psnr(predicted_tensor*255, target_tensor*255)

        # ssim_score = ssim(predicted_tensor, target_tensor)
        ssim_score1 = ssim(predicted_tensor*255, target_tensor*255)
        lpips_score=lpips_alex.forward(predicted_tensor*255, target_tensor*255).detach()


        print(f"IOU: {IOU_score:.5f}")
        print(f"SSIM1: {ssim_score1:.4f}")
        print(f"lpips: {lpips_score.item():.5f}")
        print(f"psnr: {psnr_score:.5f}")
        # print(f"psnr1: {psnr_score1:.5f}")

        IOU_list.append(IOU_score)
        ssim_list.append(ssim_score1)
        lpips_list.append(lpips_score.item())
        psnr_list.append(psnr_score)

        print('///////////////')

    if not os.path.exists(os.path.join( root_path, 'stats')):
        # Create the directory if it does not exist
        os.mkdir(os.path.join( root_path, 'stats'))

    csv_file = os.path.join( os.path.join( root_path, 'stats'),names[j]+'_'+ type_select+'_data.csv')
    print(csv_file)

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


    data = zip(IOU_list,ssim_list, lpips_list, psnr_list)

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['IOU','ssim', 'lpips', 'psnr'])
        writer.writerows(data)


    print(ssim_mean,lpips_mean,psnr_mean)
