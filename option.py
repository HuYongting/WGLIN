import argparse

def args_parse():
    parser = argparse.ArgumentParser(description='grade')
    parser.add_argument('--image-size', type=int, default=224, help='image size')


    # model
    parser.add_argument('--depth', type=int, default=12, help='')
    parser.add_argument('--head', type=int, default=9, help='')
    parser.add_argument('--in_channels', type=int, default=4, help='')
    parser.add_argument('--num_classes', type=int, default=5, help='')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='')
    parser.add_argument('--pretrained', type=bool, default=False, help='')
    parser.add_argument('--pre_Path', type=str, default="", help='')

    # train
    parser.add_argument('--lr', type=int, default=1e-5, help='')
    parser.add_argument('--epoch', type=int, default=110, help='')
    parser.add_argument('--batch-size', type=int, default=24, help='')

    # choice
    parser.add_argument('--server', type=str,  default='huyt', choices=['huyt','01','02'])
    parser.add_argument('--data-mode', type=str, default='process', choices=['process','rgb','gray'],help='默认rgb ')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--k', type=int, default=0, help='gpu id')


    parser.add_argument('--desc', type=str, default="Ablation", help='gpu id',choices=['Ablation','MV','BSV'])
    parser.add_argument('--model_name', type=str, default="WMIMVDR", help='gpu id')


    args = parser.parse_args()
    return args

