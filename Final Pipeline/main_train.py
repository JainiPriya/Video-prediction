import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')
import wandb

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir_pretrain', default='./results_pretrain', type=str)
    parser.add_argument('--res_dir_finetune', default='./results_finetune', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--pretrain_root', default='/scratch/ss16912/data_final/dataset/frames')
    parser.add_argument('--finetune_root', default='/scratch/ss16912/data_final/dataset/masks')
    parser.add_argument('--train_root', default='/scratch/ss16912/data_final/dataset/train')
    parser.add_argument('--val_root', default='/scratch/ss16912/data_final/dataset/val')
    parser.add_argument('--unlabeled_root', default='/scratch/ss16912/data_final/dataset/unlabeled')
    parser.add_argument('--dataname', default='moving_objects', choices=['moving_objects'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape_pretrain', default=[11,3,160,240], type=int,nargs='*')
    parser.add_argument('--in_shape_finetune', default=[11,1,160,240], type=int,nargs='*')
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=512, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    return parser




def pretrain_module():
    args = create_parser().parse_args()
    config = args.__dict__

    wandb.init(project="frame-pred-simvp", config=config)


    exp = Exp(args, wandb.config, pretrain=True)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start pretraining <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args, pretrain=True)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # wandb.log({"test_final_mse": mse})

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    wandb.finish()

def finetune_module():
    args = create_parser().parse_args()
    config = args.__dict__

    wandb.init(project="mask-pred-simvp", config=config)


    exp = Exp(args, wandb.config, pretrain=False)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start finetuning <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args, pretrain=False)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # wandb.log({"test_final_mse": mse})

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    wandb.finish()


if __name__ == '__main__':

    sweep_configuration = {
        "method": "random",
        "metric": { 
            "name": "vali_loss",
            "goal": "minimize"
        },
        "parameters":{
            "lr": {"values": [1e-3, 1e-2]}
            }
        }

    # Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='frame-pred-simvp', 
        )

    wandb.agent(sweep_id, function=pretrain_module, count=2)

    sweep_configuration = {
        "method": "random",
        "metric": { 
            "name": "vali_loss",
            "goal": "minimize"
        },
        "parameters":{
            "lr": {"values": [1e-3, 1e-2]}
            }
        }

    # Start the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='mask-pred-simvp', 
        )

    wandb.agent(sweep_id, function=finetune_module, count=2)


