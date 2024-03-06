# DE-RSM
A depth map enhancement model for training within the supervision of RGB and low-quality depth GT.

![image](https://github.com/dangdang17/DE-RSM/assets/78062148/da8c9d28-7367-48f2-904f-81159ac3ecfa)

Fig. 1. An example of depth GT in NYUv2. (a) RGB, (b) GT, (c) enhanced GT by our model. (d) edges of (a) and (b). (e) edges of (a) and (c). The edges of depth GT in red and RGB in white are misaligned in (d) while they are well consistent in (e).

## Run
Create a new folder named 'models' and download the [pretrained weights](https://drive.google.com/drive/folders/1o3vboZ20PhnOxLFP7U6trWJEdrJG8n3d?hl=zh_CN) inside the folder.

The requirements of environment are Python==3.8, Pytorch==2.0.

Run the following code for single depth map enhancement.

```python test_enhance_realscenes.py```

More testset can be found [here](https://github.com/Wang-xjtu/G2-MonoDepth).

Training code is coming soon.
