import argparse
import logging
import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["MASTER_ADDR"]="10.119.81.14"
#os.environ["MASTER_PORT"]="42069"
#os.environ["NODE_RANK"]="0"

import random
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch

from openfold.config_SS import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    DummyDataLoader,
)
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.argparse import remove_arguments
from openfold.utils.loss import AlphaFoldLoss, lddt_ca, compute_drmsd
from openfold.utils.seed import seed_everything
from openfold.utils.tensor_utils import tensor_tree_map, dict_multimap
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint
)

from openfold.utils.import_weights import (
    import_jax_weights_,
)

from openfold.utils.logger import PerformanceLoggingCallback
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import pdist, squareform


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config, model, output_name, train_dir_name):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = model #AlphaFold(config)
        
        #PSH ... 240703... for Tensorboard..
        self.output_name = output_name
        self.loss = AlphaFoldLoss(config.loss, self.get_writer(self.current_epoch))
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None
        
    def get_writer(self, epoch):
        logdir = os.path.join(self.config.training.tensorboard_logdir,  f"{self.output_name}_epoch_{epoch}")
        return SummaryWriter(logdir)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)
        
        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss = self.loss(outputs, batch, batch_idx)
        self.log("loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            self.cached_weights = self.model.state_dict()
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        # Calculate validation loss
        outputs = self.model(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        
        loss = lddt_ca(
                    outputs["final_atom_positions"],
                    batch["all_atom_positions"],
                    batch["all_atom_mask"],
                    eps=self.config.globals.eps,
                    per_residue=False,
                )

        self.log("val_loss", loss, prog_bar=True,on_step=True,logger=True)
        return {"val_loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.writer.add_scalar('Loss/train', avg_loss, self.current_epoch)

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.writer.add_scalar('Loss/val', avg_val_loss, self.current_epoch)
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def configure_optimizers(self, 
        learning_rate: float = 5 * 1e-4,
        eps: float = 1e-8            
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        learning_rate = self.config.training.learning_rate
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        
    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema"])
        
    def on_epoch_start(self):
        self.writer = self.get_writer(self.current_epoch)
        self.loss = AlphaFoldLoss(self.config.loss, self.writer)



def main(args):
    
    if(args.seed is not None):
        seed_everything(args.seed) 

    #PSH modifying.. for taking gpu number..
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_number
    
    config = model_config(
        args.model_name, 
        train=True, 
        low_prec=(args.precision == 16)
    ) 
    

    model = AlphaFold(config)   
    
    if config.training.use_pretrained_parameter == True:
        import_jax_weights_(model, "openfold/resources/params/params_model_5_ptm.npz", version="model_5_ptm")

    model_module = OpenFoldWrapper(config, model, args.output_name, args.train_feature_dir.split('/')[-3])


    if(args.resume_from_ckpt and args.resume_model_weights_only):
        sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
        sd = {k[len("module."):]:v for k,v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")

    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )


    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    mc = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename=args.output_name+ '_' + args.train_feature_dir.split('/')[-3] + "_{epoch}_{val_loss:.4f}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    callbacks.append(mc)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val_loss",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="max",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)
        
    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            # for taking model name to performance_log.json...
            log_file=os.path.join(args.output_dir, f"{args.model_name}_performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.deepspeed_config_path is not None):
        if "SLURM_JOB_ID" in os.environ:
            cluster_environment = SLURMEnvironment()
        else:
            cluster_environment = None
        strategy = DeepSpeedStrategy(config=args.deepspeed_config_path,cluster_environment=cluster_environment,)
    elif (args.gpus is not None and args.gpus) > 1 or args.num_nodes > 1:
        strategy = pl.strategies.DDPStrategy
    else:
        strategy = None
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs = config.training.max_epochs,
        accumulate_grad_batches=config.training.batch_size
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )



def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
        
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="model name to train"
    )
    
    parser.add_argument(
        "--train_by_features", type=bool_type, default=True,
        help="Setup the train data type. If you train with feature file, then it will be True."
    )
    
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="output"
    )
    
    parser.add_argument(
        "--train_feature_dir", type=str,
        help="Directory containing train_feature"
    )
    
    parser.add_argument(
        "--val_feature_dir", type=str,
        help="Directory containing val_feature"
    )

    parser.add_argument(
        "--GPU_number", type=str, default=None,
        help="a number of GPU"
    )
        
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/home/bis/anaconda3/envs/PSH_alphalink/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_mapping_path", type=str, default=None,
        help='''Optional path to a .json file containing a mapping from
                consecutive numerical indices to sample names. Used to filter
                the training set'''
    )
    parser.add_argument(
        "--distillation_mapping_path", type=str, default=None,
        help="""See --train_mapping_path"""
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help="""Path to obsolete.dat file containing list of obsolete PDBs and 
             their replacements."""
    )
    
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_best_val", type=bool_type, default=True,
        help="""Whether to save the model parameters that perform best during
                validation"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of them model"
    )
    
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(parser, ["--accelerator", "--resume_from_checkpoint"]) 

    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    main(args)
