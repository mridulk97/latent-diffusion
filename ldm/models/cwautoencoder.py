import os
import torch
import itertools
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F

# from contextlib import contextmanager

# from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from ldm.modules.diffusionmodules.model import Encoder, Decoder
# from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModel, AutoencoderKL
from ldm.models.disentanglement.iterative_normalization import IterNormRotation as cw_layer
from ldm.analysis_utils import get_CosineDistance_matrix, aggregatefrom_specimen_to_species
from ldm.plotting_utils import plot_heatmap_at_path


from ldm.util import instantiate_from_config

CONCEPT_DATA_KEY = "concept_data"

class CWmodelVQGAN(VQModel):


    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]


        super().__init__(**args)
        
        if not self.cw_module_infer:
            self.encoder.norm_out = cw_layer(self.encoder.block_in)
            print("Changed to cw layer after loading base VQGAN")


    def training_step(self, batch, batch_idx, optimizer_idx):
        if (batch_idx+1)%30==0 and optimizer_idx==0:
            print('cw module')
            self.eval()
            with torch.no_grad():                    
                for _, concept_batch in enumerate(self.concepts.train_dataloader()):
                    for idx, concept in enumerate(concept_batch['class'].unique()):
                        concept_index = concept.item()
                        self.encoder.norm_out.mode = concept_index
                        X_var = concept_batch['image'][concept_batch['class'] == concept]
                        X_var = X_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                        X_var = torch.autograd.Variable(X_var).cuda()
                        X_var = X_var.float()
                        self(X_var)
                        break

                self.encoder.norm_out.update_rotation_matrix()

                self.encoder.norm_out.mode = -1
            self.train()

        # breakpoint()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=False)

        # if optimizer_idx == 0 or (not self.loss.has_discriminator):
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        # if optimizer_idx == 1 and self.loss.has_discriminator:
        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
        
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        h = self.encoder(x)
        h = self.quant_conv(h)
        class_label = batch['class']

        return {'z_cw': h,
                'label': class_label,
                'class_name': batch['class_name']}

    # NOTE: This is kinda hacky. But ok for now for test purposes.
    def set_test_chkpt_path(self, chkpt_path):
        self.test_chkpt_path = chkpt_path

    @torch.no_grad()
    def test_epoch_end(self, in_out):
        postfix_name = 'inference_false'
        z_cw =torch.cat([x['z_cw'] for x in in_out], 0)
        labels =torch.cat([x['label'] for x in in_out], 0)
        sorting_indices = np.argsort(labels.cpu())
        sorted_zq_cw = z_cw[sorting_indices, :]

        classnames = list(itertools.chain.from_iterable([x['class_name'] for x in in_out]))
        sorted_class_names_according_to_class_indx = [classnames[i] for i in sorting_indices]
        z_size = sorted_zq_cw.shape[-1]
        channels = sorted_zq_cw.shape[1]
        # breakpoint()
        figs_folder = os.path.join('/', *self.test_chkpt_path.split('/')[:-2], 'figs/testset_agg')
        if not os.path.exists(figs_folder):
            os.makedirs(figs_folder)
        


        sorted_zq_cw_aggregated = aggregatefrom_specimen_to_species(sorted_class_names_according_to_class_indx, sorted_zq_cw, z_size, channels)
        z_cosine_distances = get_CosineDistance_matrix(sorted_zq_cw_aggregated)

        plot_heatmap_at_path(z_cosine_distances.cpu(), figs_folder, self.test_chkpt_path, title=f'Cosine_distances_{postfix_name}', postfix='testset_agg')

        

        z_cosine_distancess_np = z_cosine_distances.cpu().numpy()
        df = pd.DataFrame(z_cosine_distancess_np)
        df = df.drop(columns=[5, 6])
        df = df.drop([5, 6])
        breakpoint()
        path_to_save = os.path.join(figs_folder, f'CW_z_cosine_distances_{postfix_name}.csv')
        print("saved to path : ", path_to_save)
        df.to_csv(path_to_save)
        
        return None

class CWmodelInterface(VQModel):

    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]


        super().__init__(**args)
        
        if not self.cw_module_infer:
            self.encoder.norm_out = cw_layer(self.encoder.block_in)
            print("Changed to cw layer after loading base VQGAN")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class CWmodelKL(AutoencoderKL):
    def __init__(self, **args):
        print(args)
        
        self.save_hyperparameters()

        concept_data_args = args[CONCEPT_DATA_KEY]
        print("Concepts params : ", concept_data_args)
        self.concepts = instantiate_from_config(concept_data_args)
        self.concepts.prepare_data()
        self.concepts.setup()
        del args[CONCEPT_DATA_KEY]


        super().__init__(**args)
        
        if not self.cw_module_infer:
            self.encoder.norm_out = cw_layer(self.encoder.block_in)
            print("Changed to cw layer after loading base KL Autoecoder")


    def training_step(self, batch, batch_idx, optimizer_idx):
        if (batch_idx+1)%30==0 and optimizer_idx==0:
            print('cw module')
            self.eval()
            with torch.no_grad():                    
                for _, concept_batch in enumerate(self.concepts.train_dataloader()):
                    for idx, concept in enumerate(concept_batch['class'].unique()):
                        concept_index = concept.item()
                        self.encoder.norm_out.mode = concept_index
                        X_var = concept_batch['image'][concept_batch['class'] == concept]
                        X_var = X_var.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                        X_var = torch.autograd.Variable(X_var).cuda()
                        X_var = X_var.float()
                        self(X_var)
                        break

                self.encoder.norm_out.update_rotation_matrix()

                self.encoder.norm_out.mode = -1
            self.train()

        # breakpoint()
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss
