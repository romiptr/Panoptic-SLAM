# Modified by romiptr from: https://github.com/iit-DLSLab/Panoptic-SLAM/blob/main/src/python/panoptic_python.py
# Usage Debug: python3 yoso_python.py --config [PATH.yaml][OPTIONAL] --img [IMG_PATH]

import cv2
import numpy as np
from easydict import EasyDict
import yaml
import argparse

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.projects.deeplab import add_deeplab_config

from predictor import VisualizationDemo
from Mask2Former.mask2former.config import add_maskformer2_config

class Mask2Former_Net:
    def __init__(self,config):

        with open(config,'r') as file:
            yaml_content = yaml.safe_load(file)

        args = self.setup_args(yaml_content)
        cfg = self.setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

        self.output_img = None
        self.union_instance_mask = None
        self.image = None       
    #----------------------------------------------
    def setup_args(self,model_path,cfg_file):
        args_cfg = EasyDict()

        args_cfg.config_file = yaml_content['CONFIG']
        args_cfg.opts = []
        args_cfg.opts.append("MODEL.WEIGHTS")
        args_cfg.opts.append(model_path)

        return args_cfg
    #-------------------------------------------------
    def setup_cfg(self,args):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)

        cfg.freeze()
        return cfg
    #---------------------------------------------
    def mask2former_run(self,image_cv):
        predictions, visualized_output = self.demo.run_on_image(image_cv)
        return predictions,visualized_output
    #---------------------------------------------
    def mask2former_run_cpp(self,image_cv):
        self.image = image_cv

        results = []
        self.all_masks = []
        predictions, visualized_output = self.demo.run_on_image(image_cv)
        self.output_img = visualized_output.get_image()[:, :, ::-1]

        panoptic_mask_map, segments_info = predictions['panoptic_seg']
        panoptic_mask_map_np = panoptic_mask_map.cpu().numpy()

        self.union_instance_mask = np.zeros(panoptic_mask_map_np.shape, dtype=np.uint8)

        for segment in segments_info:
            tmp = []

            binary_mask = (panoptic_mask_map_np == segment['id'])
            pred_mask = (binary_mask*255).astype('uint8') 

            tmp.append(segment['id'])
            tmp.append(segment['isthing'])
            tmp.append(segment.get('score', 0.0))
            tmp.append(segment['category_id'])
            tmp.append(segment.get('instance_id', 0))
            tmp.append(segment['area'])
            
            if segment['isthing']:
                self.union_instance_mask[binary_mask] = 255
            
            tmp.append(self.binary_mask_2_bytearray(pred_mask)) # mask
            
            # bounding box
            if segment['isthing'] and segment['area'] > 0:
                rows = np.any(binary_mask, axis=1)
                cols = np.any(binary_mask, axis=0)
                if rows.any() and cols.any():
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
                else:
                    bbox = [0.0, 0.0, 1.0, 1.0] # Placeholder for empty mask
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]  # Placeholder for stuff
            tmp.append(bbox)

            results.append(tmp)
            self.all_masks.append(pred_mask)
            
        return results
    #---------------------------------------------
    def get_output_img(self):
        return bytearray(self.output_img)
    #---------------------------------------------
    def get_all_instance_mask(self):
        temp = np.zeros_like(self.image)
        v = Visualizer(temp, None, scale=1,instance_mode=2)   
        mask = v.draw_binary_mask(self.union_instance_mask,color='white',alpha=1)
        return bytearray(mask.get_image())
    #---------------------------------------------
    def get_all_masks(self,w=640,h=480):
        masks = np.zeros([h,w,3],dtype=np.uint8)
        for mask in self.all_masks:
            dst = cv2.merge((mask,mask,mask))
            masks = cv2.addWeighted(masks,1,dst,1,0)
        return bytearray(masks)
    #---------------------------------------------

    def binary_mask_2_bytearray(self,binary_mask):
        dst = cv2.merge((binary_mask,binary_mask,binary_mask))
        return bytearray(dst)
#=============================DEBUG====================================
def main():
    #print("Debug Mask2Former_Net")

    #parsing arguments--------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/Mask2Former_SLAM.yaml", help="Path to config(.yaml) file")
    parser.add_argument("--img",type=str, required=True, help="image input path")
    args=parser.parse_args()
    #-------------------------------------------

    net = Mask2Former_Net(args.config)
    
    img = cv2.imread(args.img)
    results = net.mask2former_run_cpp(img)
    output_img = net.get_output_img()
    instance_mask = net.get_all_instance_mask()

    # predictions, vis_output = net.mask2former_run(args.img)
    #print(predictions)
    #print(type(predictions)

    #WINDOW_NAME = "Panoptic visualization"
    #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #cv2.imshow(WINDOW_NAME,  output_img)
    #cv2.imshow("instance mask",instance_mask)
    #cv2.waitKey(0)
    #cv2.imwrite("panoptic_output.png",visualized_output.get_image()[:, :, ::-1])
    
    #np.savetxt("panoptic_mask.txt",predictions['panoptic_seg'][0].cpu().numpy())

    #print("done")
#===================================================
if __name__=="__main__": main()
