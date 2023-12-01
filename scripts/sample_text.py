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
    opt = parser.parse_args()

    # --scale 1.0 --n_samples 3 --ddim_steps 20

    # vq f8 64 , dim 612, trainable 
    # config_path = '/globalscratch/mridul/ldm/test/test_ldm_understand/2023-11-03T19-01-05_test_ldm_test/configs/2023-11-03T19-01-05-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_ldm_understand/2023-11-03T19-01-05_test_ldm_test/checkpoints/epoch=000146.ckpt'

    # 32 channels trainable, dim 512
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_ldm_understand/2023-11-03T22-51-04_test_ldm_test_channels32_maxlen20_dim512/checkpoints/epoch=000131.ckpt'
    # config_path = '/globalscratch/mridul/ldm/test/test_ldm_understand/2023-11-03T22-51-04_test_ldm_test_channels32_maxlen20_dim512/configs/2023-11-03T22-51-04-project.yaml'
    
    # non-trainable
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_ldm_understand/2023-11-04T04-13-39_test_ldm_test_channels32_maxlen20_dim512_cond_nottrainable/checkpoints/epoch=000131.ckpt'
    # config_path = '/globalscratch/mridul/ldm/test/test_ldm_understand/2023-11-04T04-13-39_test_ldm_test_channels32_maxlen20_dim512_cond_nottrainable/configs/2023-11-04T04-13-39-project.yaml'

    #### node 32
    # config_path = '/globalscratch/mridul/ldm/test/test_node/2023-11-05T01-39-26_test_node_channels32_maxlen77_dim640_node/configs/2023-11-05T01-39-26-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_node/2023-11-05T01-39-26_test_node_channels32_maxlen77_dim640_node/checkpoints/epoch=000131.ckpt'

    #### node 64
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-05T01-43-56_test_node_channels64_maxlen77_dim640_node/checkpoints/epoch=000215.ckpt'
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-05T01-43-56_test_node_channels64_maxlen77_dim640_node/configs/2023-11-05T01-43-56-project.yaml'


    #### node 32 - CLIP base
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T11-41-05_CLIP_channels32_maxlen77_node/configs/2023-11-07T11-41-05-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T11-41-05_CLIP_channels32_maxlen77_node/checkpoints/epoch=000149.ckpt'

    #### node 32 - CLIP scale 0.18
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T12-11-42_CLIP_channels32_maxlen77_node_scale0.18215/configs/2023-11-07T12-11-42-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T12-11-42_CLIP_channels32_maxlen77_node_scale0.18215/checkpoints/epoch=000221.ckpt'

    #### node 32 - CLIP scale 0.18 SD
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T12-16-07_CLIP_channels32_maxlen77_node_scale0.18215_SD/configs/2023-11-07T12-16-07-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T12-16-07_CLIP_channels32_maxlen77_node_scale0.18215_SD/checkpoints/epoch=000317.ckpt'

    #### node 32 - CLIP base batch 4
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T21-47-01_CLIP_channels32_maxlen77_node_batch4/configs/2023-11-07T21-47-01-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T21-47-01_CLIP_channels32_maxlen77_node_batch4/checkpoints/epoch=000158.ckpt'

    #### node 32 - CLIP base batch 1
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T22-00-45_CLIP_channels32_maxlen77_node_batch1/configs/2023-11-07T22-00-45-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T22-00-45_CLIP_channels32_maxlen77_node_batch1/checkpoints/epoch=000143.ckpt'

    # #### node 32 - CLIP scale 0.5
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T12-10-02_CLIP_channels32_maxlen77_node_scale0.5/configs/2023-11-07T12-10-02-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-07T12-10-02_CLIP_channels32_maxlen77_node_scale0.5/checkpoints/epoch=000215.ckpt'

    # #### node 32 - CLIP scale 2.0
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-08T21-59-32_CLIP_channels32_maxlen77_node_batch8_scale2.0/configs/2023-11-08T21-59-32-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-08T21-59-32_CLIP_channels32_maxlen77_node_batch8_scale2.0/checkpoints/epoch=000131.ckpt'

    # #### node 32 - CLIP scale 5.0
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-08T21-59-32_CLIP_channels32_maxlen77_node_batch8_scale5.0/configs/2023-11-08T21-59-32-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-08T21-59-32_CLIP_channels32_maxlen77_node_batch8_scale5.0/checkpoints/epoch=000131.ckpt'

    # #### node 32 - CLIP max_len 30
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-08T22-04-53_CLIP_channels32_maxlen77_node_batch8_lr5e7_maxlen30/configs/2023-11-08T22-04-53-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-08T22-04-53_CLIP_channels32_maxlen77_node_batch8_lr5e7_maxlen30/checkpoints/epoch=000188.ckpt'

    # #### CLIP f4
    # config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/configs/2023-11-09T15-34-23-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/checkpoints/epoch=000158.ckpt'
    
    # #### CLIP f8
    # config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-30-05_CLIP_f8_maxlen77_classname/configs/2023-11-09T15-30-05-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-30-05_CLIP_f8_maxlen77_classname/checkpoints/epoch=000119.ckpt'

    # #### BERT f4 node weighted 256
    config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_BERT_f4_node_weighted_nospecial_token_max256/configs/2023-11-13T23-08-55-project.yaml'
    ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_BERT_f4_node_weighted_nospecial_token_max256/checkpoints/epoch=000119.ckpt'



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
    # prompt = '1, 2, 5, 11, 14, 23'

    # class_to_node = '/fastscratch/mridul/fishes/class_to_node_bfs.pkl'
    class_to_node = '/fastscratch/mridul/fishes/class_to_node_bfs_weighted.pkl'
    with open(class_to_node, 'rb') as pickle_file:
        class_to_node_dict = pickle.load(pickle_file)
        

    # sample_path = os.path.join(outpath, "samples_nodes_clip32_scale0.18215")
    sample_path = os.path.join(outpath, opt.output_dir_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    # artificial_class_names = ['Gambusia Notropis', 'Gambusia Notorous', 'Gambusia Lepomis', 'Gambusia Morone',
                            #    'Gambusia Esox', 'Gambusia Cyprinus', 'Gambusia Alosa', 'Gambusia Lepisosteus']
    # artificial_class_names = ['Notropis Gambusia', 'Notorous Gambusia', 'Lepomis Gambusia', 'Morone Gambusia',
    #                            'Esox Gambusia', 'Cyprinus Gambusia', 'Alosa Gambusia', 'Lepisosteus Gambusia']

    breakpoint()

    for class_name, node_representation in tqdm(class_to_node_dict.items()):
    # for class_name in tqdm(artificial_class_names):
        prompt = node_representation
        # prompt = class_name

        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    all_prompts = opt.n_samples * [prompt]
                    print(all_prompts)
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    # c1 = model.get_learned_conditioning(opt.n_samples * [prompt1])
                    # c2 = model.get_learned_conditioning(opt.n_samples * [prompt2])
                    # c = c1*0.4 + c2*0.6
                    # shape = [4, opt.H//8, opt.W//8]
                    # shape = [4, 32, 32]
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

                    # for x_sample in x_samples_ddim:
                    #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    #     # Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    #     base_count += 1
                    all_samples.append(x_samples_ddim)


        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{class_name.replace(" ", "-")}.png'))
        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{prompt2.replace(" ", "-")+ " " + opt.postfix}.png'))



    print(f"Your samples are ready and waiting four you here: \n{sample_path} \nEnjoy.")


# python sample_text.py --outdir /home/mridul/sample_ldm --scale 1.0 --n_samples 4 --ddim_steps 200 --ddim_eta 1.0 --output_dir_name hybrid_bert