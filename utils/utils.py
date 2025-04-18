from pathlib import Path

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math

def load_loggers(cfg, args):
    log_path = cfg.General.log_path

    path = args.path
    log_path = log_path

    Path(log_path).mkdir(exist_ok=True, parents=True)

    log_name = Path(cfg.config).parent
    version_name = Path(cfg.config).name[:-5]

    cfg.log_path = Path(log_path) / version_name / f'{path}'
    #cfg.log_path= Path(log_path) / version_name /'classic' /'
    print(f'---->Log dir: {cfg.log_path}')


    tb_logger = WandbLogger(project= 'Self-supervised-learning',
                        name= version_name,
                        log_model=True,
                        save_dir=(log_path)
                         )

    #---->CSV
    csv_logger = pl_loggers.CSVLogger(Path(log_path),
                                      name = f'{path}',
                                      version =version_name)

    return [tb_logger, csv_logger]


def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=cfg.General.patience,
    #     verbose=True,
    #     mode='min'
    # )
    # Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',

                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = -1,
                                         mode = 'min',
                                        every_n_epochs=20,
                                        save_weights_only = True))
    return Mycallbacks

from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, base_lr, final_lr):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine decay
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            min_lr_ratio = final_lr / base_lr
            return cosine_decay * (1 - min_lr_ratio) + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)
