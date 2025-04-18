import argparse
from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

import re
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters

def make_parse():

    """
    Parse command line arguments for training or testing.

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='configs/SIM_CLR.yaml',type=str)
    parser.add_argument('--gpus', default = [2])
    parser.add_argument('--path', default='train', type=str)
    args = parser.parse_args()
    return args

def extract_epoch_from_filename(filename):
    """
    Extracts the epoch number from a filename like 'epoch=020-val_loss=0.1234.ckpt'.
    Returns 0 if not found.
    """
    match = re.search(r'epoch[=_\-]?(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0
#---->main
def main(cfg):

    """
    Main function to run training or testing based on stage.

    Args:
        cfg (Namespace): Configuration object loaded from YAML
    """

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg, args)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}

    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                           'epochs': cfg.General.epochs,
                            'lr': cfg.Optimizer.lr,
                            'path': args.path
                            }

    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus,
        precision=cfg.General.precision,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1
    )

    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        #model_paths = [str(model_path) for model_path in model_paths if 'last' in str(model_path)]

        for path in model_paths:
            print(f"Testing: {path}")
            epoch = extract_epoch_from_filename(Path(path).name)
            print (epoch)

            # Load model
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)

            # Set a safe custom attribute for plotting
            new_model._epoch_for_plot = epoch

            # Run test (automatically triggers test_epoch_end and UMAP plot)
            trainer.test(model=new_model, datamodule=dm)



if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage


    main(cfg)
 