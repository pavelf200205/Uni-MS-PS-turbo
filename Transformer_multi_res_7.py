import torch.nn as nn
import torch
import numpy as np
from Transformer_8 import Transformer_8
from utils_process import decrease_size_batch
from tqdm import tqdm


    
class Transformer_multi_res_7(nn.Module):
    
    def __init__(self,
                 c_in=3,
                 eval_mode=False,
                 patch_size=256,
                 padding=32,
                 overlap=32,
                 initial_stage_number=4,
                 batch_size_encoder=3,
                 batch_size_transformer=5000):
        
        super(Transformer_multi_res_7, self).__init__()
        
        self.eval_mode = eval_mode
        self.Net_first = Transformer_8(c_in=c_in,
                                       dim_hidden=[64, 128, 256, 512],
                                       eval_mode=eval_mode,
                                       batch_size_encoder=batch_size_encoder,
                                       batch_size_transformer=batch_size_transformer)
        c_in+=3
        self.Net_stage = Transformer_8(c_in=c_in,
                                       dim_hidden=[64, 128, 256, 512],
                                       eval_mode=eval_mode,
                                       batch_size_encoder=batch_size_encoder,
                                       batch_size_transformer=batch_size_transformer)
        
        self.c_in      = c_in
        self.inference_mode = False
            
        
        self.batch_size_encoder = batch_size_encoder
        self.batch_size_transformer = batch_size_transformer
        
        
        self.initial_stage_number = initial_stage_number
        self.patch_size = patch_size
        self.padding = padding
        self.overlap = overlap
        
        self.stride = self.patch_size-2*self.padding-2*self.overlap
        
        
        
        self.layer_unfold = torch.nn.Unfold(self.patch_size,
                                            dilation=1,
                                            padding=0,
                                            stride=self.stride)
           
    def set_inference_mode(self, use_cuda_eval_mode=False):
        self.inference_mode = True
        self.Net_first.set_inference_mode(use_cuda_eval_mode=use_cuda_eval_mode)
        self.Net_stage.set_inference_mode(use_cuda_eval_mode=use_cuda_eval_mode)
        
    def prepareInputs(self, x, nb_stage, stage_number):
        imgs = torch.moveaxis(x["imgs"], 2, 0)
        
        mask = x["mask"]
        mask = ~mask
        
        inputs = []
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            img = imgs[i].contiguous()
            inputs.append(img)
      
        stage_number = nb_stage-stage_number
        import torch.nn.functional as F
        for i in range(1, stage_number):
            
            temp = []
            for j in range(len(inputs)):
                res = F.interpolate(inputs[j], scale_factor=0.5, mode='bilinear', align_corners=False)
                temp.append(res)
            
            inputs = temp
      
            mask = F.interpolate(mask.float(), scale_factor=0.5, mode='nearest') > 0.5
            
        return inputs, mask
    
    def build_unfold(self, img):

        img1 = self.layer_unfold(img.float())
        img1 =  img1.reshape(img.shape[0],
                             img.shape[1],
                             self.patch_size,
                             self.patch_size,
                             img1.shape[-1])
     
        return img1
    
    def build_unfold_img(self, img, coord_x, coord_y):
        img = img[:,:,
                  coord_x,
                  coord_y]

        img =  img.reshape(img.shape[0],
                           img.shape[1],
                           self.patch_size,
                           self.patch_size)
     
        return img


    def build_fold(self, img, size_img, coords_x, coords_y):
        
        result = torch.zeros(img.shape[0],
                             img.shape[1],
                             size_img[0],
                             size_img[1], device=img.device, dtype=img.dtype)
        
        for i in range(img.shape[-1]):
            result[:,:,
                   coords_x[:,:,i],
                   coords_y[:,:,i]] += img[:,:,:,:,i]

        return result    
    

    def find_size_stage(self, num_stage,
                        nb_stage,
                        stride=768,
                        patch_size=1024):
        
        size = int(32*(2**num_stage))
        decrease_step = nb_stage - (num_stage+1)
        return size, size, decrease_step    
    
    
    def interpolate_normal(self, normal, shape):
        normal = torch.nn.functional.interpolate(input=normal,
                                                 size=shape,
                                                 align_corners=True,
                                                 mode="bilinear")
        normal = torch.nn.functional.normalize(normal, 
                                               2, 1) 
        return normal
    
    def gen_mask_patch(self, coord_x, coord_y, shape_img):
        mask_patch = torch.ones((1, 3,
                                 self.patch_size,
                                 self.patch_size), device=coord_x.device)
        border = False
        
        c_min_x = torch.min(coord_x).item()
        c_max_x = torch.max(coord_x).item()
        c_min_y = torch.min(coord_y).item()
        c_max_y = torch.max(coord_y).item()

        if c_min_x==0 and c_min_y!=0:
            mask_patch[:,:,
                       :-self.padding,
                       self.padding:-self.padding] = 0
            border = True
        elif c_min_x==0 and c_min_y==0:
            mask_patch[:,:,
                       :-self.padding,
                       :-self.padding] = 0
            border = True
        elif c_min_x!=0 and c_min_y==0:
            mask_patch[:,:,
                       self.padding:-self.padding,
                       :-self.padding] = 0
            border = True
            
        if c_max_x==shape_img[0]-1 and c_max_y!=shape_img[1]-1:
            mask_patch[:,:,
                       self.padding:,
                       self.padding:-self.padding] = 0
            border = True
        elif c_max_x==shape_img[0]-1 and c_max_y==shape_img[1]-1:
            mask_patch[:,:,
                       self.padding:,
                       self.padding:] = 0
            border = True
        elif c_max_x!=shape_img[0]-1 and c_max_y==shape_img[1]-1:
            mask_patch[:,:,
                       self.padding:-self.padding,
                       self.padding:] = 0
            border = True
            
        if c_min_x==0 and c_max_y==shape_img[1]-1:
            mask_patch[:,:,
                       :-self.padding,
                       self.padding:] = 0
            border = True
            
        if c_max_x==shape_img[0]-1 and c_min_y==0:
            mask_patch[:,:,
                       self.padding:,
                       :-self.padding:] = 0
            border = True
            
        if not border:
            mask_patch[:,:,
                       self.padding:-self.padding,
                       self.padding:-self.padding] = 0
        return mask_patch
    
    def gen_weight_normal_mask(self):
        if hasattr(self, '_mask_weight_cache'):
            return self._mask_weight_cache
        ax = np.linspace(-(self.patch_size - 1) / 2.,
                         (self.patch_size - 1) / 2.,
                         self.patch_size)
        sig = self.patch_size/5
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        weight = kernel / np.sum(kernel)
        weight = torch.from_numpy(weight).half()
        self._mask_weight_cache = weight
        return weight
    
    def forward(self, x, nb_stage):
        inputs, masks = self.prepareInputs(x)
        
        pred = {}
        pred["others_scale_n"] = []
        pred["others_scale_n_error"] = []
        normal = None
        
        for i in range(nb_stage):
            normal = self.forward_stage(imgs=inputs[i],
                                        mask=masks[i],
                                        index_scale=i,
                                        normal=normal)
            stage_pred = {"n":normal}
            if i<self.nb_stage-1:
                pred["others_scale_n"].append(normal)
        
        masks[-1] = masks[-1].to(normal.device)
        pred['n'] = nn.functional.normalize(stage_pred["n"], 2, 1)
        pred['n'] = pred['n'].masked_fill(masks[-1], 0) 
        return pred
    
    
    def forward_stage(self, imgs, mask, index_scale, normal=None):
        for j in range(len(imgs)):
            if index_scale>0 and j==0: 
                normal = nn.functional.interpolate(input=normal.detach(),
                                                   size=imgs[j][0,0].shape,
                                                   align_corners=True,
                                                   mode="bilinear")
                normal = nn.functional.normalize(normal, 2, 1)

            if index_scale>0:
                normal = normal.to(imgs[j].device)
                imgs[j] = torch.cat([imgs[j], normal], 1)

        temp = torch.stack(imgs).permute(1,0,2,3,4)

        if index_scale==0:
            stage_pred = self.Net_first.forward([temp,
                                                 mask])
        else:
            stage_pred = self.Net_stage.forward([temp,
                                                 mask])
            
        normal = stage_pred["n"]
        normal = nn.functional.normalize(normal, 2, 1)
        return normal
    
    
    
    def process(self, x, nb_stage):
        batch_size_patches = 1  # Reduced from 8 to 4 to fit in 12GB VRAM
        
        # Pre-calculate total steps for a unified progress bar
        total_steps = self.initial_stage_number
        for i in range(self.initial_stage_number, nb_stage):
            size_stage_x, size_stage_y, decrease_step = self.find_size_stage(num_stage=i, nb_stage=nb_stage, stride=self.stride, patch_size=self.patch_size)
            x1 = torch.arange(0, size_stage_x)
            y1 = torch.arange(0, size_stage_y)
            coords = torch.meshgrid(x1, y1, indexing='ij')
            c_x = coords[0].unsqueeze(0).unsqueeze(0).float()
            c_x = self.build_unfold(c_x).long().squeeze()
            num_patches = c_x.shape[-1]
            total_steps += (num_patches + batch_size_patches - 1) // batch_size_patches
            
        pbar = tqdm(total=total_steps, desc="Overall Inference")
        
        normal = None
        
        for i in range(nb_stage):
            inputs, masks = self.prepareInputs(x,
                                               nb_stage=nb_stage,
                                               stage_number=i)

            device = next(self.parameters()).device
            inputs = [inp.to(device) for inp in inputs]
            masks = masks.to(device)
            if normal is not None:
                normal = normal.to(device)

            temps_imgs = []
            if i<self.initial_stage_number:
                normal = self.forward_stage(imgs=inputs,
                                            mask=masks,
                                            index_scale=i,
                                            normal=normal)
                pbar.update(1)

            else:
                size_stage_x, size_stage_y, decrease_step = self.find_size_stage(num_stage=i,
                                                                                 nb_stage = nb_stage,
                                                                                 stride=self.stride,
                                                                                 patch_size=self.patch_size)
   
                self.size_img_pad = (size_stage_x,
                                     size_stage_y)                
                
                normal = self.interpolate_normal(normal=normal,
                                                 shape=[size_stage_x,
                                                        size_stage_y])
                normal = torch.nn.functional.normalize(normal, 2, 1)  

                x1 = torch.arange(0, self.size_img_pad[0])
                y1 = torch.arange(0, self.size_img_pad[1])
                coords = torch.meshgrid(x1, y1,
                                        indexing='ij')
                coords_x = coords[0].unsqueeze(0).unsqueeze(0).float()
                coords_y = coords[1].unsqueeze(0).unsqueeze(0).float()
                coords_x = self.build_unfold(coords_x).long().squeeze()
                coords_y = self.build_unfold(coords_y).long().squeeze()
                
                coords_x_dev = coords_x.to(device)
                coords_y_dev = coords_y.to(device)
                
                normal_output = torch.zeros(1, 3,
                                            self.patch_size,
                                            self.patch_size,
                                            coords_x.shape[-1], device=device, dtype=torch.float16)
                num_patches = coords_x.shape[-1]
                for j_start in range(0, num_patches, batch_size_patches):
                    j_end = min(j_start + batch_size_patches, num_patches)
                    K = j_end - j_start
                    temps_imgs = []
                    for k in range(len(inputs)):
                        with torch.no_grad():
                            img = inputs[k]
                            img_extracted = img[:, :, coords_x_dev[:,:,j_start:j_end], coords_y_dev[:,:,j_start:j_end]]
                            img_extracted = img_extracted.permute(0, 4, 1, 2, 3).reshape(-1, img.shape[1], self.patch_size, self.patch_size)
                            temps_imgs.append(img_extracted)
                    
                    normal1_extracted = normal[:, :, coords_x_dev[:,:,j_start:j_end], coords_y_dev[:,:,j_start:j_end]]
                    normal1 = normal1_extracted.permute(0, 4, 1, 2, 3).reshape(-1, normal.shape[1], self.patch_size, self.patch_size)
                    with torch.no_grad():
                        masks_list = []
                        for k_idx in range(K):
                            masks_list.append(self.gen_mask_patch(coord_x=coords_x_dev[:,:,j_start+k_idx],
                                                                  coord_y=coords_y_dev[:,:,j_start+k_idx],
                                                                  shape_img=self.size_img_pad))
                        mask_patches = torch.cat(masks_list, dim=0)
                        
                        mask_patches = mask_patches.to(masks.device)
                        
                        masks_extracted = masks[:, :, coords_x_dev[:,:,j_start:j_end], coords_y_dev[:,:,j_start:j_end]]
                        masks_extracted = masks_extracted.permute(0, 4, 1, 2, 3).reshape(-1, masks.shape[1], self.patch_size, self.patch_size)
                        
                        mask_patches = (mask_patches * masks_extracted) > 0
                        
                        normal1 = self.forward_stage(imgs=temps_imgs,
                                                    mask=mask_patches,
                                                    index_scale=i,
                                                    normal=normal1)
                        
                    mask_weight = self.gen_weight_normal_mask()
                    mask_weight = mask_weight.to(normal1.device)
             
                    for k_idx in range(K):
                        normal_output[:,:,
                                      :,:,
                                      j_start + k_idx] += (normal1[k_idx:k_idx+1] * mask_weight)
                    
                    del normal1, normal1_extracted, mask_patches, masks_extracted, temps_imgs
                                      
                    pbar.update(1)
             
                normal = self.build_fold(normal_output,
                                         coords_x = coords_x_dev,
                                         coords_y = coords_y_dev,
                                         size_img=self.size_img_pad)   
            
            torch.cuda.empty_cache()
        
        pbar.close()
        masks = masks.to(normal.device)
        normal = torch.nn.functional.normalize(normal, 2, 1) 
        normal = normal.masked_fill(masks, 0) 
        return {"n": normal}


    def load_weights(self,
                     file):

        checkpoint = torch.load(file, 
                                map_location=torch.device('cpu'))
      
        self.load_state_dict(checkpoint) 
