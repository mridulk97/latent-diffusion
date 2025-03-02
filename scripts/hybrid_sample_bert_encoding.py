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

    # # #### CLIP f4
    # config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/configs/2023-11-09T15-34-23-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-34-23_CLIP_f4_maxlen77_classname/checkpoints/epoch=000158.ckpt'
    
    # # #### CLIP f8
    # config_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-30-05_CLIP_f8_maxlen77_classname/configs/2023-11-09T15-30-05-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/clip/2023-11-09T15-30-05_CLIP_f8_maxlen77_classname/checkpoints/epoch=000119.ckpt'

    #### Label Encoding
    # config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_TEST_f4_ancestral_label_encoding/configs/2023-11-13T23-08-55-project.yaml'
    # ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_TEST_f4_ancestral_label_encoding/checkpoints/epoch=000119.ckpt'

    # #### BERT f4 node weighted 256
    config_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_BERT_f4_node_weighted_nospecial_token_max256/configs/2023-11-13T23-08-55-project.yaml'
    ckpt_path = '/globalscratch/mridul/ldm/test/test_bert/2023-11-13T23-08-55_BERT_f4_node_weighted_nospecial_token_max256/checkpoints/epoch=000119.ckpt'


    label_to_class_mapping = {0: 'Alosa-chrysochloris', 1: 'Carassius-auratus', 2: 'Cyprinus-carpio', 3: 'Esox-americanus', 
    4: 'Gambusia-affinis', 5: 'Lepisosteus-osseus', 6: 'Lepisosteus-platostomus', 7: 'Lepomis-auritus', 8: 'Lepomis-cyanellus', 
    9: 'Lepomis-gibbosus', 10: 'Lepomis-gulosus', 11: 'Lepomis-humilis', 12: 'Lepomis-macrochirus', 13: 'Lepomis-megalotis', 
    14: 'Lepomis-microlophus', 15: 'Morone-chrysops', 16: 'Morone-mississippiensis', 17: 'Notropis-atherinoides', 
    18: 'Notropis-blennius', 19: 'Notropis-boops', 20: 'Notropis-buccatus', 21: 'Notropis-buchanani', 22: 'Notropis-dorsalis', 
    23: 'Notropis-hudsonius', 24: 'Notropis-leuciodus', 25: 'Notropis-nubilus', 26: 'Notropis-percobromus', 
    27: 'Notropis-stramineus', 28: 'Notropis-telescopus', 29: 'Notropis-texanus', 30: 'Notropis-volucellus', 
    31: 'Notropis-wickliffi', 32: 'Noturus-exilis', 33: 'Noturus-flavus', 34: 'Noturus-gyrinus', 35: 'Noturus-miurus', 
    36: 'Noturus-nocturnus', 37: 'Phenacobius-mirabilis'}

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

    class_to_node = '/fastscratch/mridul/fishes/class_to_node_bfs_weighted.pkl'
    with open(class_to_node, 'rb') as pickle_file:
        class_to_node_dict = pickle.load(pickle_file)
    
    # breakpoint()
    # hybrid_dict = {'Gambusia affinis': ['1, 3, 3, 4'], 'Gambusia75buchanani25': ['1, 3, 3, 21'], 
    #                'Gambusia50buchanani50': ['1, 3, 7, 21'], 'Gambusia25buchanani75': ['1, 1, 7, 21'],
    #                'Notropis buchanani': ['0, 1, 7, 21']}
    
    # hybrid_dict = {'Carassius auratus': ['0, 1, 1, 1'], 'Carassius75buchanani25': ['0, 1, 1, 21'],
    #                'buchanani75carassius25': ['0, 1, 7, 1'],
    #                'Notropis buchanani': ['0, 1, 7, 21']}
    hybrid_dict = {'Noturus nocturnus': ['0, 5, 8, 36'], 'Noturus-to-buchanani_1': ['0, 5, 8, 21'],
                   'Noturus-to-buchanani_2': ['0, 5, 7, 21'], 'Notropis buchanani': ['0, 1, 7, 21'], 
                   'buchanani-to-Noturus_1': ['0, 1, 7, 36'], 'buchanani-to-Noturus_2': ['0, 1, 8, 36'],}
    

    # Lepomis gulosus = '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88 <N> 24 <E> 16.78 <N> 41'
    lepomis_auritus = ['<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88 <N> 24 <E> 2.83 <N> 40 <E> 13.95 <N> 50',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88 <N> 24 <E> 2.83 <N> 40 <E> 13.95',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88 <N> 24 <E> 2.83 <N> 40',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88 <N> 24 <E> 2.83',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88 <N> 24',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14 <E> 125.88',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0 <N> 14',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11 <E> 0.0',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94 <N> 11',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5 <E> 78.94',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77 <N> 5',
                       '<N> 1 <E> 80.54 <N> 2 <E> 10.77',
                       '<N> 1 <E> 80.54 <N> 2',
                       '<N> 1 <E> 80.54',
                       '<N> 1'
    ]

    # 'Lepisosteus platostomus': '<N> 1 <E> 270.16 <N> 3 <E> 42.75 <N> 6', 'Lepisosteus osseus': '<N> 1 <E> 270.16 <N> 3 <E> 42.75 <N> 7'

    notropis_buchanani = ['<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48 <N> 12 <E> 28.95 <N> 16 <E> 109.35 <N> 34',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48 <N> 12 <E> 28.95 <N> 16 <E> 109.35',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48 <N> 12 <E> 28.95 <N> 16',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48 <N> 12 <E> 28.95',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48 <N> 12',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4',
            '<N> 1 <E> 80.54 <N> 2 <E> 29.69',
            '<N> 1 <E> 80.54 <N> 2',
            '<N> 1 <E> 80.54',
            '<N> 1',
            ]
    
    # (Pdb) class_to_node_dict['Notropis buccatus']
# '<N> 1 <E> 80.54 <N> 2 <E> 29.69 <N> 4 <E> 25.9 <N> 8 <E> 38.48 <N> 12 <E> 28.95 <N> 16 <E> 71.98 <N> 37 <E> 4.07 <N> 42 <E> 4.07 <N> 51 <E> 13.83 <N> 54 <E> 4.88 <N> 55 <E> 10.53 <N> 58'

    sample_path = os.path.join(outpath, opt.output_dir_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    breakpoint()

    # for class_name, node_representation in tqdm(class_to_node_dict.items()):
    for i in range(len(notropis_buchanani)):
    # for class_name, node_representation in tqdm(hybrid_dict.items()):
        prompt = notropis_buchanani[i]
        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
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

        ###### to make grid
        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{class_name.replace(" ", "-")}.png'))
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'{str(i)}.png'))

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



# {'Alosa chrysochloris': ['0, 0, 0, 0'], 'Carassius auratus': ['0, 1, 1, 1'], 'Cyprinus carpio': ['0, 1, 1, 2'], 
#  'Esox americanus': ['1, 2, 2, 3'], 'Gambusia affinis': ['1, 3, 3, 4'], 'Lepisosteus osseus': ['2, 4, 4, 5'], 
#  'Lepisosteus platostomus': ['2, 4, 4, 6'], 'Lepomis auritus': ['1, 3, 5, 7'], 'Lepomis cyanellus': ['1, 3, 5, 8'], 
#  'Lepomis gibbosus': ['1, 3, 5, 9'], 'Lepomis gulosus': ['1, 3, 5, 10'], 'Lepomis humilis': ['1, 3, 5, 11'], 
#  'Lepomis macrochirus': ['1, 3, 5, 12'], 'Lepomis megalotis': ['1, 3, 5, 13'], 'Lepomis microlophus': ['1, 3, 5, 14'], 
#  'Morone chrysops': ['1, 3, 6, 15'], 'Morone mississippiensis': ['1, 3, 6, 16'], 'Notropis atherinoides': ['0, 1, 7, 17'], 
#  'Notropis blennius': ['0, 1, 7, 18'], 'Notropis boops': ['0, 1, 7, 19'], 'Notropis buccatus': ['0, 1, 7, 20'], 
#  'Notropis buchanani': ['0, 1, 7, 21'], 'Notropis dorsalis': ['0, 1, 7, 22'], 'Notropis hudsonius': ['0, 1, 7, 23'], 
#  'Notropis leuciodus': ['0, 1, 7, 24'], 'Notropis nubilus': ['0, 1, 7, 25'], 'Notropis percobromus': ['0, 1, 7, 26'], 
#  'Notropis stramineus': ['0, 1, 7, 27'], 'Notropis telescopus': ['0, 1, 7, 28'], 'Notropis texanus': ['0, 1, 7, 29'], 
#  'Notropis volucellus': ['0, 1, 7, 30'], 'Notropis wickliffi': ['0, 1, 7, 31'], 'Noturus exilis': ['0, 5, 8, 32'], 
#  'Noturus flavus': ['0, 5, 8, 33'], 'Noturus gyrinus': ['0, 5, 8, 34'], 'Noturus miurus': ['0, 5, 8, 35'], 
#  'Noturus nocturnus': ['0, 5, 8, 36'], 'Phenacobius mirabilis': ['0, 1, 7, 37']}
