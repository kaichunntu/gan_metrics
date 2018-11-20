## Metrics
-------------------------------------------------
### Setup
Download the metric code from [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans)
and put them into current dir which contains metric.py(i.e. ms_ssim.py and metric.py must be under same dir).

### Usage

    python metric.py $1 $2 -m -s
    $1: gan_img_dir
    $2: ground_truth_img_dir
    
Note : Both director must have same count of image.


### Requirement
Please see requirements.txt or use command "pip install -r requirements.txt".


### Reference 
1. [Progressive GAN](https://github.com/tkarras/progressive_growing_of_gans)