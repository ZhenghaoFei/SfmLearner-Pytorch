import torch

from imageio import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_float
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import models
from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--dispnet_type', default='single', metavar='STR',
                    help='dispnet type, single: current frame (from original code) '
                    'triple: use frame n, n+1, n-1 as input for dispnet (to capture parallax from motion)')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return
    if args.dispnet_type == 'single':
        disp_net = models.DispNetS().to(device)
    elif args.dispnet_type == 'triple':
        disp_net = models.DispNetSTri(nb_ref_imgs=2).to(device)

    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])
    test_files = sorted(test_files)
    print('{} files to test'.format(len(test_files)))
    # print('files to test', test_files)


    # for file in tqdm(test_files):
    for idx in tqdm(range(len(test_files))):

        file = test_files[idx]

        if args.dispnet_type == 'triple':
            if idx == 0 or idx == len(test_files) - 1:
                continue
            tgt_img  = load_img_to_tensor(args, test_files[idx])
            ref_img  = load_imgs_to_tensor(args, [test_files[idx-1], test_files[idx+1]])
            output = disp_net(tgt_img, ref_img)[0]
        else:
            tgt_img  = load_img_to_tensor(args, file)
            output = disp_net(tgt_img)[0]



        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall()[1:])

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))

def load_img_to_tensor(args, file):
    img = img_as_float(imread(file))

    h,w,_ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = resize(img, (args.img_height, args.img_width))
    img = np.transpose(img, (2, 0, 1))

    tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    tensor_img = ((tensor_img - 0.5)/0.5).to(device)
    return tensor_img

def load_imgs_to_tensor(args, files):
    images = []
    for file in files:
        img = img_as_float(imread(file))
        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = resize(img, (args.img_height, args.img_width))
        img = np.transpose(img, (2, 0, 1))
        images.append(img)
    images = np.concatenate(images, axis=0)
    images = torch.from_numpy(images.astype(np.float32)).unsqueeze(0)
    images = ((images - 0.5)/0.5).to(device)
    return images

if __name__ == '__main__':
    main()
