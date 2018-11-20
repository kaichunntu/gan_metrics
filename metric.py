
import os
import cv2
import glob
import argparse
import numpy as np
import sys
from sliced_wasserstein import API as swd
from ms_ssim import API as ms_ssim


eval_count = 100

def MY_PARSER():
    parser = argparse.ArgumentParser(prog="Metric function for gan image.",
                                     description="Calculate MSSIM and SWD. Image format must be jpg and count of generated/ground truth image must be multiple of 100 and equal.")
    parser.add_argument("gan_path",action="store",type=str,help="Folder path for generative image")
    parser.add_argument("gt_path",action="store",type=str,help="Folder path for ground truth image")
    parser.add_argument("-m","--mssim",action="store_true",dest="mssim_flag",default=False,
                        help="Use for calculating MS-SSIM")
    parser.add_argument("-s","--swd",action="store_true",dest="swd_flag",default=False,
                        help="Use for calculating sliced wasserstein distance")
    return parser

def get_data(gan_path , gt_path):
    gan_img_path = glob.glob(os.path.join(gan_path,'*.jpg'))
    ground_img_path = glob.glob(os.path.join(gt_path,'*.jpg'))
    gan_img_path = gan_img_path[0:len(gan_img_path) - (len(gan_img_path)%eval_count)]
    ground_img_path = ground_img_path[0:len(ground_img_path) - (len(ground_img_path)%eval_count)]
    assert len(gan_img_path) == len(ground_img_path)

    print('Prepare gan data')
    gan_images = []
    for i in gan_img_path:
        image = cv2.imread(i, cv2.IMREAD_COLOR)
        gan_images.append(cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA))
    gan_images = np.array(gan_images)
    gan_images = gan_images.transpose(0,3,1,2)
    print('Load gan data finish')

    print('Prepare target data')
    ground_images = []
    for i in ground_img_path:
        image = cv2.imread(i, cv2.IMREAD_COLOR)
        ground_images.append(cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA))
    ground_images = np.array(ground_images)
    ground_images = ground_images.transpose(0,3,1,2)
    print('Load target data finisth')
    
    return gan_images , ground_images



def calculate_MSSSIM(gan_images,ground_images):
    print('MS-SSIM evaluate')
    ms = ms_ssim(gan_images.shape[0],gan_images.shape[2:],gan_images.dtype,eval_count)
    ms.begin('warmup')
    for i in range(0,len(gan_images),eval_count):
        ms.feed('warmup',gan_images[i:i+eval_count])
        print("Counts : " , i , end="\r")
    score = ms.end('warmup')
    print('MS-SSIM finish')
    ms.end('fake')
    return score




def calculate_SWD(gan_images,ground_images):
    print('SWD evaluate')
    sswd = swd(gan_images.shape[0],gan_images.shape[2:],gan_images.dtype,eval_count)
    sswd.begin('warmup')
    for i in range(0,len(ground_images),eval_count):
        sswd.feed('warmup',ground_images[i:i+eval_count])
        print('ground:' + str(i), end="\r")
    _ = sswd.end('warmup')
    sswd.begin('fakes')
    for i in range(0,len(gan_images),eval_count):
        sswd.feed('fakes',gan_images[i:i+eval_count])
        print('fakes:' + str(i), end="\r")
    score = sswd.end('fakes')
    print('SWD finisth')
    return score




if __name__ == "__main__":
    parser = MY_PARSER()
    args = parser.parse_args()
    gan_images , gt_images = get_data(args.gan_path,args.gt_path)
    if args.mssim_flag:
        ms_score = calculate_MSSSIM(gan_images,gt_images)
    if args.swd_flag:
        sswd_score = calculate_SWD(gan_images,gt_images)
    if args.mssim_flag:
        print('MS-SSIM score',ms_score)
    if args.swd_flag:
        print('SWD score',sswd_score)




