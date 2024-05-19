import argparse, os, sys, glob
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# import colored_traceback.always


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    # pl_sd = torch.load(ckpt, map_location="cpu")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        # default=0.0,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        # default=4,
        default=20,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--output_dir_name",
        type=str,
        default='default_file',
        help="name of folder",
    )

    parser.add_argument(
        "--postfix",
        type=str,
        default='',
        help="name of folder",
    )

    parser.add_argument(
        "--scale",
        type=float,
        # default=5.0,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    # --scale 1.0 --n_samples 3 --ddim_steps 20

    # dim 768
    ckpt_path = '/fastscratch/mridul/new_diffusion_models/ldm/cub_conditioning/2024-01-31T21-25-36_HLE_f4_epoch185_batch8_lr5e-6/checkpoints/epoch=000113.ckpt'
    config_path = '/fastscratch/mridul/new_diffusion_models/ldm/cub_conditioning/2024-01-31T21-25-36_HLE_f4_epoch185_batch8_lr5e-6/configs/2024-01-31T21-25-36-project.yaml'



    def get_label_from_class(class_name):
        for key, value in label_to_class_mapping.items():
            if value == class_name:
                return key

    config = OmegaConf.load(config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, ckpt_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    all_images = []
    labels = []

    # class_to_node = '/projects/ml4science/mridul/ldm/data/cub/cub_class_to_ancestral_level.pkl'
    # with open(class_to_node, 'rb') as pickle_file:
    #     class_to_node_dict = pickle.load(pickle_file)


    hybrid_dict = {'clark_nutcracker': ['7, 9, 48, 88'],
                    'clar_nutcracker_l2_loggerhead_shrike': ['7, 31, 48, 88'], 'clar_nutcracker_l2_vireo': ['7, 34, 48, 88'],
                    'clark_nutcracker_l3_crow': ['7, 9, 16, 88'],
                    }
    
    # breakpoint()

    sample_path = os.path.join(outpath, opt.output_dir_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # for class_name, node_representation in tqdm(class_to_node_dict.items()):
    for class_name, node_representation in tqdm(hybrid_dict.items()):
        prompt = node_representation
        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    xc = opt.n_samples * (prompt)
                    xc = [tuple(xc)]
                    print(class_name, prompt)
                    c = model.get_learned_conditioning({'class_to_node': xc})
                    shape = [3, 64, 64]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    all_samples.append(x_samples_ddim)

        # ###### to make grid
        # # additionally, save as grid
        # grid = torch.stack(all_samples, 0)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # grid = make_grid(grid, nrow=opt.n_samples)

        # # to image
        # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{class_name.replace(" ", "-")}.png'))

        # individual images
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')


        # class_number = int(prompt[0].split(',')[ancestry_level-1])
        # class_name = f'class_{class_number}'
        level = class_name.split('_')[0]
        # save_dir_path = os.path.join(sample_path, level, class_name)
        save_dir_path = os.path.join(sample_path, class_name)
        os.makedirs(save_dir_path, exist_ok=True)
        for i in range(opt.n_samples):
            sample = grid[i]
            img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
            img_arr = img.astype(np.uint8)
            Image.fromarray(img_arr).save(f'{save_dir_path}/{class_name}_img_{i}.png')

    #     # additionally, save as grid
    #     grid = torch.stack(all_samples, 0)
    #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')

    #     for i in range(opt.n_samples):
    #         sample = grid[i]
    #         img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
    #         img_arr = img.astype(np.uint8)
    #         class_name = class_name.replace(" ", "-")
    #         all_images.append(img_arr)
    #         labels.append(get_label_from_class(class_name))
    #         Image.fromarray(img_arr).save(f'{sample_path}/{class_name}_{i}.png')

    # all_images = np.array(all_images)
    # labels = np.array(labels)

    # np.savez(sample_path + '.npz', all_images, labels)


    print(f"Your samples are ready and waiting four you here: \n{sample_path} \nEnjoy.")


# python sample_text.py --outdir /home/mridul/sample_images_text --scale 1.0 --n_samples 3 --ddim_steps 200 --ddim_eta 1.0
