import requests
import os
import argparse
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(_ROOT_DIR))

def apply_lama(im_path, mask_path, save_path):
  r = requests.post(
    'https://clipdrop-api.co/cleanup/v1',
    files={
      'image_file': (os.path.basename(im_path), open(im_path, 'rb'), 'image/jpeg'),
      'mask_file': (os.path.basename(mask_path), open(mask_path, 'rb'), 'image/png')
    },
    headers={
      'x-api-key': '649f32d262bc88fdc58249c908bef9601611ccb32565177364fbcfea63af0291ef383fa83505acb549d318d1a4d99293'
    }
  )
  if r.ok:
    with open(save_path, 'wb') as f:
      f.write(r.content)
  else:
    r.raise_for_status()

def main():
  parser = argparse.ArgumentParser(description='Apply LaMa inpainting using ClipDrop API.')
  parser.add_argument('--base_path', type=str, default='output_simple/recons', help='Base directory to search for images (processes all subdirs ending with "512"). If not set, use --image, --mask, and --output for single image processing.')

  args = parser.parse_args()

  root_path = _ROOT_DIR.parent
  abs_path = os.path.join(root_path, args.base_path)


  for dirpath, dirnames, filenames in os.walk(abs_path):
    if dirpath[-3:] == '512':
      im_path = os.path.join(dirpath, 'texture_interpolation_simple.png')
      mask_path = args.mask_default or args.mask
      save_path = os.path.join(dirpath, 'texture_interpolation_simple_lama.png')
      if os.path.exists(im_path) and os.path.exists(mask_path):
        print(f'Processing {im_path} with mask {mask_path}...')
        apply_lama(im_path, mask_path, save_path)
      else:
        print(f'Skipping {im_path} or mask {mask_path} (not found)')

if __name__ == '__main__':
  main()