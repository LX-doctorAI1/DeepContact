from __future__ import print_function
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def parse_opt():
    parser = argparse.ArgumentParser(description='预测数据')
    parser.add_argument("-m", "--model", default='cell',
                        help="tissue, cell, tem")
    parser.add_argument("-mm", "--mitomodel", default='',
                        help="Mito模型路径")
    parser.add_argument("-em", "--ermodel", default='',
                        help="ER模型的路径")
    parser.add_argument("--er_model_type", default='fpn')
    parser.add_argument("-lm", "--ldmodel", default='',
                        help="LD模型的路径")
    parser.add_argument("-d", "--datadir", required=False,
                        default='data/tissues/json/test/20210111_3Label_ER-Mito-LD',
                        help='要处理的数据集JSON文件保存位置')
    parser.add_argument("-o", "--outputdir", required=False,
                        default='./results',
                        help='数据输出文件夹')
    parser.add_argument("-reso", "--resolution", required=False,
                        default=5.0, type=float,
                        help='图像分辨率')
    parser.add_argument('--subset', required=False, default='',
                        help='train, test, val')
    parser.add_argument('--threshold', default=0.3, help='')
    parser.add_argument('--px', default=10, help='')

    parser.add_argument('--gpu', required=False,
                        default='0,1,2,3', help='train, test, val')
    parser.add_argument('--repeat_num', required=False, default=5, help='')
    parser.add_argument('--visall', required=False, default=False, action='store_true', help='')

    # config
    parser.add_argument('--cfg', type=str, default=None,
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from .config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # Check if args are valid
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return args


def add_eval_options(parser):
    pass

def add_diversity_opts(parser):
    pass


# Sampling related options
def add_eval_sample_opts(parser):
    pass


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0]]
    args = parse_opt()
    print(args)
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml']
    args1 = parse_opt()
    print(dict(set(vars(args1).items()) - set(vars(args).items())))
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml', '--visual_extractor', 'densenet']
    args2 = parse_opt()
    print(dict(set(vars(args2).items()) - set(vars(args1).items())))