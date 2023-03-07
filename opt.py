import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir',
                        type=str,
                        required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap'],
                        help='which dataset to train/test')
    parser.add_argument('--split',
                        type=str,
                        default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample',
                        type=float,
                        default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='scene scale (whole scene must lie in [-scale, scale]^3')

    parser.add_argument('--half2_opt',
                        action='store_true',
                        default=False,
                        help='whether to use half2 optimization')

    # loss parameters
    parser.add_argument('--distortion_loss_w',
                        type=float,
                        default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size',
                        type=int,
                        default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy',
                        type=str,
                        default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument(
        '--random_bg',
        action='store_true',
        default=False,
        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--val_only',
                        action='store_true',
                        default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test',
                        action='store_true',
                        default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name',
                        type=str,
                        default='exp',
                        help='experiment name')
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='pretrained checkpoint to load (including optimizers, etc)')

    parser.add_argument(
        '--gui',
        action='store_true',
        default=False,
        help='whether to show interactive GUI after training is done'
    )

    # performance profile
    parser.add_argument('--perf', action='store_true', default=False)

    return parser.parse_args()
