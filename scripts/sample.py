import torch
from omegaconf import OmegaConf
import os
from tqdm import tqdm

from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-09-30T15-19-56_VQ_batch8-lr1e6-channels_32/configs/2023-09-30T15-19-56-project.yaml")  
    model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-09-30T15-19-56_VQ_batch8-lr1e6-channels_32/checkpoints/epoch=000149.ckpt")
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-04T21-26-56_VQ_batch8-lr1e6-channels_32-postquant/configs/2023-10-04T21-26-56-project.yaml")  
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-04T21-26-56_VQ_batch8-lr1e6-channels_32-postquant/checkpoints/epoch=000038.ckpt")
    # post quant conv
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-13T18-22-21_LDM_batch8-lr1e6-channels_32-postquant_conv/configs/2023-10-13T18-22-21-project.yaml")  
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-13T18-22-21_LDM_batch8-lr1e6-channels_32-postquant_conv/checkpoints/epoch=000131.ckpt")
    # post quant
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-13T01-36-31_LDM_batch8-lr1e6-channels_32-postquant/configs/2023-10-13T01-36-31-project.yaml")  
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-13T01-36-31_LDM_batch8-lr1e6-channels_32-postquant/checkpoints/epoch=000131.ckpt")

    # ######3 f8
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm/2023-10-23T18-07-25_vqf8_gpus2_lr1e6_batch16/configs/2023-10-23T18-07-25-project.yaml")  
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm/2023-10-23T18-07-25_vqf8_gpus2_lr1e6_batch16/checkpoints/epoch=000149.ckpt")

    #### f8 64 cw
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-27T20-17-09_CW_batch16-lr1e6-channels64_epoch17/configs/2023-10-27T20-17-09-project.yaml")  
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-27T20-17-09_CW_batch16-lr1e6-channels64_epoch17/checkpoints/epoch=000224.ckpt")


    ##### 128 channels
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-24T13-11-06_VQ_batch16-lr1e6-channels_128_epoch275/configs/2023-10-24T13-11-06-project.yaml")
    # model = load_model_from_config(config, "//fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-24T13-11-06_VQ_batch16-lr1e6-channels_128_epoch275/checkpoints/epoch=000167.ckpt")

    # #### 128 pre conv
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-25T22-22-27_VQ_batch16-lr1e6-channels_128_re/configs/2023-10-25T22-22-27-project.yaml")
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-25T22-22-27_VQ_batch16-lr1e6-channels_128_re/checkpoints/epoch=000302.ckpt")

    # #### 64 pre conv
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-25T22-24-47_VQ_batch16-lr1e6-channels64_epoch245_re/configs/2023-10-25T22-24-47-project.yaml")
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm-vq/2023-10-25T22-24-47_VQ_batch16-lr1e6-channels64_epoch245_re/checkpoints/epoch=000224.ckpt")



    #### 4 channels f8
    # config = OmegaConf.load("/fastscratch/mridul/new_diffusion_models/ldm/ldm/2023-09-06T21-09-18_ldm_f8_againepoch_92/configs/2023-09-06T21-09-18-project.yaml")
    # model = load_model_from_config(config, "/fastscratch/mridul/new_diffusion_models/ldm/ldm/2023-09-06T21-09-18_ldm_f8_againepoch_92/checkpoints/epoch=000065.ckpt")

    # ## imagenet
    # config = OmegaConf.load("/globalscratch/mridul/ldm/test/test_f4/2023-10-18T19-22-34_test_ldm_f4_lr1e6_batch32/configs/2023-10-18T19-22-34-project.yaml")  
    # model = load_model_from_config(config, "/globalscratch/mridul/ldm/test/test_f4/2023-10-18T19-22-34_test_ldm_f4_lr1e6_batch32/checkpoints/epoch=000017.ckpt")  
    return model

from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


classes = range(38)   # define classes to be sampled here
# classes = [25, 2] 
n_samples_per_class = 80

ddim_steps = 200
ddim_eta = 1.0
scale = 1.0   # for unconditional guidance

all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {'class': torch.tensor(n_samples_per_class*[37]).to(model.device)}
            )
        for class_label in tqdm(classes):
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
            c = model.get_learned_conditioning({'class': xc.to(model.device)})
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                            #  shape=[16, 16, 16],
                                             shape=[32, 32, 32],
                                            #  shape=[3, 64, 64], # f4
                                            #  shape=[4, 32, 32], #f8
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)


# # display as grid
# grid = torch.stack(all_samples, 0)
# grid = rearrange(grid, 'n b c h w -> (n b) c h w')
# grid = make_grid(grid, nrow=n_samples_per_class)

# # to image
# grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
# grid_pil = Image.fromarray(grid.astype(np.uint8))

# file_path = '/home/mridul/output_image.png'  # Specify the desired file path
# grid_pil.save(file_path)


label_to_class_mapping = {0: 'Alosa-chrysochloris', 1: 'Carassius-auratus', 2: 'Cyprinus-carpio', 3: 'Esox-americanus', 
4: 'Gambusia-affinis', 5: 'Lepisosteus-osseus', 6: 'Lepisosteus-platostomus', 7: 'Lepomis-auritus', 8: 'Lepomis-cyanellus', 
9: 'Lepomis-gibbosus', 10: 'Lepomis-gulosus', 11: 'Lepomis-humilis', 12: 'Lepomis-macrochirus', 13: 'Lepomis-megalotis', 
14: 'Lepomis-microlophus', 15: 'Morone-chrysops', 16: 'Morone-mississippiensis', 17: 'Notropis-atherinoides', 
18: 'Notropis-blennius', 19: 'Notropis-boops', 20: 'Notropis-buccatus', 21: 'Notropis-buchanani', 22: 'Notropis-dorsalis', 
23: 'Notropis-hudsonius', 24: 'Notropis-leuciodus', 25: 'Notropis-nubilus', 26: 'Notropis-percobromus', 
27: 'Notropis-stramineus', 28: 'Notropis-telescopus', 29: 'Notropis-texanus', 30: 'Notropis-volucellus', 
31: 'Notropis-wickliffi', 32: 'Noturus-exilis', 33: 'Noturus-flavus', 34: 'Noturus-gyrinus', 35: 'Noturus-miurus', 
36: 'Noturus-nocturnus', 37: 'Phenacobius-mirabilis'}

npz_output_file = '/home/mridul/sample_ldm/individual_images/class_conditional_steps200_eta1_samples80.npz'
dir_path = '/home/mridul/sample_ldm/individual_images/class_conditional_steps200_eta1_samples80'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

all_images = []
labels = []
j=0

# for class_label in classes:
#     for i in range(n_samples_per_class):
#         sample = all_samples[j][i]
#         img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
#         img_arr = Image.fromarray(img.astype(np.uint8))
#         breakpoint()
#         img_arr.save(f'{dir_path}/image_{i}_{label_to_class_mapping[class_label]}.png')
#     j += 1


for class_label in classes:
    for i in range(n_samples_per_class):
        sample = all_samples[j][i]
        img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
        img_arr = img.astype(np.uint8)
        all_images.append(img_arr)
        labels.append(class_label)
        Image.fromarray(img_arr).save(f'{dir_path}/{label_to_class_mapping[class_label]}_{i}.png')
        # Image.fromarray(img_arr).save(f'{dir_path}/image_{i}_{class_label}.png')
    j += 1

all_images = np.array(all_images)
labels = np.array(labels)

np.savez(npz_output_file, all_images, labels)



# sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
# sample = sample.permute(0, 2, 3, 1)
#         sample = sample.contiguous()


