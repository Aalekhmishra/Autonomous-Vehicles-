
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

## test a single image and show the results
img = 'demo.png'  # or 
img= mmcv.imread(img)
print(img)
result = inference_segmentor(model, img)
# visualize the results in a new window
print(result)
model.show_result(img, result, show=True)
 
model.show_result(img, result, out_file='result.jpg', opacity=0.5)

