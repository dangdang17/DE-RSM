import os
import copy
import torch
import datetime

import numpy as np

from PIL import Image
from networks import rzUNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    enhance()
    better_show()

def enhance():
    # enhance raw depth map in realscene
    # input: rgb_image(rgb_dir)
    #        raw_depth_map(raw_dir, ranging in [0, 65535])
    #        loaded_model(model_dir, pretrained_weight)
    # output: enhanced depth map(also ranging in [0, 65535])

    rgb_dir = 'test_images/rgb_1151.png'
    raw_dir = 'test_images/gt_1151.png'
    save_dir = 'test_images/enhance_1151.png'

    network = rzUNet()
    network = network.cuda()
    network.load_state_dict(torch.load('models/aim16_epoch_60.pth')['network_state_dict'])

    rgb_pil = Image.open(rgb_dir)
    raw_pil = Image.open(raw_dir)
    rgb_tensor = torch.from_numpy((np.array(rgb_pil, dtype=float) / 255.).transpose((2, 0, 1))).unsqueeze(0)
    raw_tensor = torch.from_numpy((np.array(raw_pil, dtype=float) / 65535.)).unsqueeze(0).unsqueeze(0) # reranging depth into [0,1]
    raw_input_tensor = torch.nn.functional.interpolate(raw_tensor, (rgb_tensor.shape[2], rgb_tensor.shape[3]))
    hole_tensor = torch.ones((1,1,rgb_tensor.shape[2],rgb_tensor.shape[3]))
    hole_tensor[raw_input_tensor==0] = 0

    rgb_tensor = rgb_tensor.to(torch.float32)
    raw_input_tensor = raw_input_tensor.to(torch.float32)
    hole_tensor = hole_tensor.to(torch.float32)

    enhance_tensor = network(rgb_tensor.cuda(), raw_input_tensor.cuda(), hole_tensor.cuda())
    enhance_array = enhance_tensor.squeeze().to('cpu').detach().numpy()
    enhance_array = np.clip(enhance_array * 65535., 0, 65535).astype(np.int32) # come from [0,1], we here use unified range of [0, 65535]
    enhance_pil = Image.fromarray(enhance_array)

    print(f'enhancement result is saved as {save_dir}')
    enhance_pil.save(save_dir)

def better_show():
    # to better show depth map
    # input: gt_depth_map
    #        enhance_depth_map
    # output: gt_rescaled_depth_map
    #         enhance_rescaled_depth_map

    gt_dir = 'test_images/gt_1151.png'
    enhance_dir = 'test_images/enhance_1151.png'

    gt_array = np.array(Image.open(gt_dir), dtype=np.float32)
    enhance_array = np.array(Image.open(enhance_dir), dtype=np.float32)
    max = gt_array.max()
    min = gt_array.min()

    gt_rescale_array = np.array((gt_array-min)/(max-min)*65535, dtype=np.int32)
    gt_rescale_pil = Image.fromarray(gt_rescale_array)
    gt_rescale_pil.save('test_images/gt_show_1151.png')
    print('result for better show is saved as test_images/gt_show_1151.png')

    enhance_rescale_array = np.array((enhance_array-min)/(max-min)*65535, dtype=np.int32)
    enhance_rescale_pil = Image.fromarray(enhance_rescale_array)
    enhance_rescale_pil.save('test_images/enhance_show_1151.png')
    print('result for better show is saved as test_images/enhance_show_1151.png')
  
if __name__ == "__main__":
    tm_begin = datetime.datetime.now()
    print('tm_begin: ', tm_begin)
    main()
    tm_end = datetime.datetime.now()
    print('tm_begin: ', tm_begin)
    print('tm_end: ', tm_end)


