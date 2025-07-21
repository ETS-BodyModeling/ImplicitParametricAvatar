import requests
import os
def apply_lama(im_path,mask_path,save_path):
  r = requests.post('https://clipdrop-api.co/cleanup/v1',
    files = {
      'image_file': ('result_female-19_512.jpg', open(im_path, 'rb'), 'image/jpeg'),
      'mask_file': ('mask1.jpg', open(mask_path, 'rb'), 'image/png')
      },
    headers = { 'x-api-key': '649f32d262bc88fdc58249c908bef9601611ccb32565177364fbcfea63af0291ef383fa83505acb549d318d1a4d99293'}
    # headers = { 'x-api-key': '9a10006ebbf05fdbacf5f1ada9e969919b10b26f1f7e340b581b8bc1007d954d663d61ed76154be85eef932dc46e548e'}
  )
  if (r.ok):
    with open(save_path, 'wb') as f:
      f.write(r.content)
  else:
    r.raise_for_status()

base_path = '/home/fares/ImplicitParametricAvatar/output/recons'
for (dirpath, dirnames, filenames) in os.walk(base_path):
    # if dirpath[-3:] == '512':
    #     apply_lama(dirpath +'/texture_interpolation_simple.png',
    #             '/home/fares/ImplicitParametricAvatar/data/combined_mask.png', 
    #             dirpath + '/texture_interpolation_simple_lama.png'
    #             )
    if dirpath[-3:] == '512':
        apply_lama(dirpath +'/texture_interpolation_simple.png',
                '/home/fares/ImplicitParametricAvatar/data/combined_mask.png', 
                dirpath + '/texture_interpolation_simple_lama.png'
                )