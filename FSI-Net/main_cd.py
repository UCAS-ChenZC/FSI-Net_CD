from argparse import ArgumentParser
import torch
from models.trainer import *

# print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    ###     VSCode tensor print setting     ###
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    ###     VSCode tensor print setting     ###
    
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args): 
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='127_SFIB117_FreRMT_FreWave-t', type=str)       #116Test_Dy_sfb_TIPDA_RMT-t   tiaoshi    124_New_RMT-t
    parser.add_argument('--checkpoint_root', default='checkpoints/20260112', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)   #   CDD  LEVIR

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="trainval", type=str)
    parser.add_argument('--split_val', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    # parser.add_argument('--pretrain', default='/home/solid/CD/code/open-cd-main/fpn_dat_t_80k.pth', type=str)
    # parser.add_argument('--pretrain', default= 'pretrain/dat_pp_tiny_in1k_224.pth', type=str)              #   for UNet
    # parser.add_argument('--pretrain', default=None, type=str)   
    parser.add_argument('--pretrain', default='pretrain/RMT-T.pth', type=str)   #pretrain/RMT-S-label.pth  pretrain/RMT-T.pth
    parser.add_argument('--multi_scale_train', default=True, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [1.0, 1.0, 0.3, 0.3, 2.0])     #多尺度训练时损失函数的加权比例 [0.5, 0.5, 0.5, 0.8, 1.0] [1.0, 1.0, 0.3, 0.3, 2.0]

    parser.add_argument('--net_G', default='FSI-Former', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | ')
    # parser.add_argument('--net_G', default='ChangeFormerV6', type=str,
    #                     help='base_resnet18 | base_transformer_pos_s4 | '
    #                          'base_transformer_pos_s4_dd8 | '
    #                          'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    parser.add_argument('--loss', default='ohem', type=str)       #dice_loss  ce      ohem      ohem503

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')                                   #STEP??
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    # print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
