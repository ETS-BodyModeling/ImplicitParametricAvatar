import cv2
import numpy as np
import os
from os import walk
import requests


def interpolate_image(image, mask):
    # Get the dimensions of the image and mask
    height, width = image.shape[:2]
    # print(height, width)
    interpolated_image = image.copy()
    # print(interpolated_image.shape)
    for y in range(height):
        start = []
        end = []
        for x in range(width):
            if mask[y, x] == 1 and len(start)==len(end) :
              if x!=0:
                start.append(x-1)
              else:
                start.append(x)
            elif mask[y, x] == 0 and len(start)==(len(end)+1)  :
                end.append(x)

        if len(end)!= len(start):
          end.append(width-1)
        if not start:
          print("The list is not empty.")
          continue


        for i in range(len(start)):
            num_pixels = end[i] - start[i]+1
            weight = np.linspace(0,1,num_pixels)
            for step in range (num_pixels):
              target_x = start[i] +  step
              interpolated_image[y, target_x] = (1 - weight[step]) * image[y, start[i]] + weight[step] * image[y, end[i]]

    return interpolated_image

def apply_lama(im_path,mask_path,save_path):
  r = requests.post('https://clipdrop-api.co/cleanup/v1',
    files = {
      'image_file': ('result_female-19_512.jpg', open(im_path, 'rb'), 'image/jpeg'),
      'mask_file': ('mask1.jpg', open(mask_path, 'rb'), 'image/png')
      },
    headers = { 'x-api-key': 'add_Lama_API_key_here' }

  )
  if (r.ok):
    with open(save_path, 'wb') as f:
      f.write(r.content)
  else:
    r.raise_for_status()

def filter_mask(mask_img):
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_img, connectivity=8)
  min_size_threshold = 500
  filtered_mask = np.zeros_like(mask_img)
  for label in range(1, num_labels):  # Exclude background component (label 0)
      area = stats[label, cv2.CC_STAT_AREA]
      if area >= min_size_threshold:
          filtered_mask[labels == label] = 255
  return filtered_mask
def create_mask(indices, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[indices] = 255
    return mask
def inpaint_interpolation(path,dir):
  image_p = cv2.imread(path, cv2.IMREAD_COLOR)
  image_p = cv2.resize(image_p, (1024, 1024), interpolation=cv2.INTER_AREA)

  indices = np.where(np.all(image_p == [0, 0, 0], axis=-1))
  image_shape = (image_p.shape[0],image_p.shape[0])
  mask_image1 = create_mask(indices, image_shape)

  # cv2.imwrite('/content/mask0.png', mask_image1)
  kernel = np.ones((5,5), dtype=np.uint8)
  mask_image1 = cv2.dilate(mask_image1, kernel, iterations=1)

  # cv2.imwrite('/content/mask1.png', mask_image1)

  interpolated_image = interpolate_image(image_p, mask_image1/255)
  cv2.imwrite(dir, interpolated_image)
  return interpolated_image,mask_image1