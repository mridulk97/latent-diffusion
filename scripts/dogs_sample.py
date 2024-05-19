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
        default=0.0,
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
        default=4,
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

    parser.add_argument(
        "--model_name",
        type=str,
        # default=5.0,
        default='hle',
        help="Name of Model",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        # default=5.0,
        default='/globalscratch/mridul/ldm/dogs/model_runs/2024-04-29T20-28-55_HLE_custom_subset/configs/2024-04-29T20-28-55-project.yaml',
        help="Path to config_file",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        # default=5.0,
        default='/globalscratch/mridul/ldm/dogs/model_runs/2024-04-29T20-28-55_HLE_custom_subset/checkpoints/epoch=000200.ckpt',
        help="Path to ckpt_path",
    )

    opt = parser.parse_args()



    def get_label_from_class(class_name):
        for key, value in label_to_class_mapping.items():
            if value == class_name:
                return key



    config = OmegaConf.load(opt.config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    ## making base paths
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, opt.output_dir_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))



    # reading class names and ndoe mapping
    if opt.model_name == 't2t' or opt.model_name == 't2t_special':
        print('Here at T2T')
        class_to_node = '/projects/ml4science/mridul/data/monkeys_dataset/monkeys_t2t_labels.pkl'
    # # elif opt.model_name == 'clip':
    # #     print('Here at CLIP')
    # #     class_to_node = '/projects/ml4science/mridul/ldm/data/cub/cub_to_scientific.pkl'
    # elif opt.model_name == 'hle_6levels':
    #     class_to_node = '/projects/ml4science/mridul/data/monkeys_dataset/monkey_HLE_labels_6levels.pkl'
    # elif opt.model_name == 'hle_v2':
    #     class_to_node = '/projects/ml4science/mridul/data/monkeys_dataset/monkey_HLE_labels_v2.pkl'
    else:
        class_to_node = '/projects/ml4science/mridul/data/stanford_dogs/dog_custom_subset_HLE_labels.pkl'
    
    with open(class_to_node, 'rb') as pickle_file:
        class_to_node_dict = pickle.load(pickle_file)

    #sorting the dictionary
    class_to_node_dict = {key: class_to_node_dict[key] for key in sorted(class_to_node_dict)}
    cub_actual_labels = sorted([key for key in class_to_node_dict.keys()])
    label_to_class_mapping = {index: value for index, value in enumerate(cub_actual_labels)}



    # training_images_list_file = '/fastscratch/mridul/cub_190_split_resized/official/CUB_200_2011/train_segmented_imagenet_background_bb_crop_256.txt'
    # with open(training_images_list_file, "r") as f:
    #     paths = sorted(f.read().splitlines())
    # class_labels = list(map(lambda path: path.split('/')[-2], paths))
    # cub_actual_labels = sorted(list(set(class_labels)))
    


    all_images = []
    labels = []


    for class_name, node_representation in tqdm(class_to_node_dict.items()):
    # for class_name in tqdm(cub_actual_labels):
        # prompt = node_representation
        if opt.model_name == 'clip_english':
            prompt = class_name.split('.')[1].replace('_', ' ')
        else:
            prompt = node_representation

        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    if opt.model_name == 'cc':
                        class_label = get_label_from_class(class_name)
                        print(f"rendering {opt.n_samples} examples of class '{class_label}' in {opt.ddim_steps} steps and using s={opt.scale:.2f}.")
                        all_prompts = torch.tensor(opt.n_samples*[class_label])
                        c = model.get_learned_conditioning({'class': all_prompts.to(model.device)})
                    elif opt.model_name in ['hle', 'hle_v2', 'hle_6levels']:
                        all_prompts = opt.n_samples * (prompt)
                        all_prompts = [tuple(all_prompts)]
                        print(class_name, prompt)
                        c = model.get_learned_conditioning({'class_to_node': all_prompts})
                    else:
                        all_prompts = opt.n_samples * [prompt]
                        print(all_prompts)
                        c = model.get_learned_conditioning(opt.n_samples * [prompt])
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



        # individual images
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')

        os.makedirs(os.path.join(sample_path, class_name), exist_ok=True)
        for i in range(opt.n_samples):
            sample = grid[i]
            img = 255. * rearrange(sample, 'c h w -> h w c').cpu().numpy()
            img_arr = img.astype(np.uint8)
            class_name = class_name.replace(" ", "-")
            all_images.append(img_arr)
            labels.append(get_label_from_class(class_name))
            Image.fromarray(img_arr).save(f'{sample_path}/{class_name}/{class_name}_{i}.png')

    all_images = np.array(all_images)
    labels = np.array(labels)

    np.savez(sample_path + '.npz', all_images, labels)

        # # additionally, save as grid
        # grid = torch.stack(all_samples, 0)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{class_name.replace(" ", "-")}.png'))
        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{prompt2.replace(" ", "-")+ " " + opt.postfix}.png'))



    print(f"Your samples are ready and waiting four you here: \n{sample_path} \nEnjoy.")


# python scripts/sample_text.py --outdir /home/mridul/sample_ldm/individual_images/ --scale 1.0 --n_samples 100 --ddim_steps 200 --ddim_eta 1.0 --output_dir_name bert_f4_512_epoch119

# python scripts/cub_sample_text.py --outdir /projects/ml4science/mridul/ldm/cub_samples --scale 1.0 --n_samples 60 --ddim_steps 200 --ddim_eta 1.0 --output_dir_name cub_clip_sample60 --model_name t2t