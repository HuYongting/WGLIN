import os
import random
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as transform
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
import models.create_models as create
from tqdm import tqdm
from dataset import CustomDataset, MultiviewImgDataset_mask, MultiviewImgDataset_no_lesion, SingleimgDataset,MultiviewImgDataset
from option import args_parse
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args,device):

    model = create.my_WMIMVDR(args,
        in_chans=args.in_channels,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        pre_Path= args.pre_Path,
        depth=args.depth,
        num_heads=args.head,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        cuda=device,
    )
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    criterion1 = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(args.epoch):

        train_epoch_loss_2 = 0.0
        train_epoch_acc_2 = 0.0

        model.train()
        train_bar = tqdm(train_loader)
        for i, (img, label) in enumerate(train_bar):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            B, V, C, H, W = img.size()
            imgs_mix = img.view(-1, C, H, W)
            evidences_ce = model(imgs_mix)
            loss_ce = criterion1(evidences_ce, label)
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_acc_2 += (evidences_ce.argmax(dim=1) == label).sum()
            train_epoch_loss_2 += loss_ce.item()

            scheduler.step(epoch + i / len(train_loader))
            optimizer.step()

        train_loss_mean_2 = train_epoch_loss_2 / len(train_loader)
        train_acc_mean_2 = train_epoch_acc_2 / (len(train_dataset) * NUM_VIEW)


        print('EPOCH:{}\t Train Loss:{:.4f}\t Train Acc:{:.3f}\t LR:{:.4e}'.format(epoch, train_loss_mean_2,
                                                                                   train_acc_mean_2,
                                                                                   optimizer.param_groups[-1]['lr']))
        logging.info('EPOCH:{}\t Train Loss:{:.4f}\t Train Acc:{:.3f}\t LR:{:.4e}'.format(epoch, train_loss_mean_2,
                                                                                          train_acc_mean_2,
                                                                                          optimizer.param_groups[-1][
                                                                                              'lr']))

    name = "DESC={},MODEL_NAME={},LR={},DEPTH={},HEAD={},EPOCH={}".format(args.desc, args.model_name,
                                                                          args.lr, args.depth, args.head,
                                                                          args.epoch)
    SAVE_PT_DIR = 'RESULTS/{}/weights'.format(name)
    torch.save(model.state_dict(), f'{SAVE_PT_DIR}/model.pth')


def raw_data(args):
    DATA_PATH = ''
    TRAIN_PATH = ''
    TEST_PATH = ''
    if args.server == 'huyt':
        DATA_PATH = "/home/huyt/DATASET_DR/fundas/EYData_BaseEye_newdata/"
    elif args.server == '01':
        DATA_PATH = "/mnt/data/huyongting/fundas/EYData_BaseEye_newdata/"
    elif args.server == '02':
        DATA_PATH = "/disk2/huyongting/fundas/EYData_BaseEye_newdata/"
    MASK_PATH = "{}train_mask".format(DATA_PATH)
    MASK_PATH2 = "{}test_mask".format(DATA_PATH)
    if args.data_mode == 'rgb':
        TRAIN_PATH = "{}train/rgb".format(DATA_PATH)
        TEST_PATH = "{}test/rgb".format(DATA_PATH)
    elif args.data_mode == 'process':
        TRAIN_PATH = "{}train_process".format(DATA_PATH)
        TEST_PATH = "{}test_process".format(DATA_PATH)
    elif args.data_mode == 'gray':
        TRAIN_PATH = "{}train/gray".format(DATA_PATH)
        TEST_PATH = "{}test/gray".format(DATA_PATH)

    # --------- 生成文件夹 -----------
    NUM_VIEW = 1
    name = "DESC={},MODEL_NAME={},LR={},DEPTH={},HEAD={},EPOCH={}".format(args.desc,args.model_name, args.lr, args.depth, args.head, args.epoch)
    SAVE_IMG_DIR = 'RESULTS/{}/imgs'.format(name)
    SAVE_PT_DIR = 'RESULTS/{}/weights'.format(name)
    LOGS_DIR = 'RESULTS/{}/LOGS'.format(name)
    os.makedirs(SAVE_IMG_DIR, exist_ok=True)
    os.makedirs(SAVE_PT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    train_csv_path = os.path.join(DATA_PATH, 'train_rgb_label_newname.csv')
    test_csv_path = os.path.join(DATA_PATH, 'test_rgb_label_newname.csv')

    return DATA_PATH, MASK_PATH, MASK_PATH2, TRAIN_PATH, TEST_PATH,train_csv_path, test_csv_path,NUM_VIEW,LOGS_DIR


if __name__ == '__main__':
    args = args_parse()
    args.k = 0

# -------------------------------------------------------------------------------------
    seed_everything(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_PATH, MASK_PATH, MASK_PATH2, TRAIN_PATH, TEST_PATH, train_csv_path, test_csv_path, NUM_VIEW, LOGS_DIR = raw_data(
        args)
    test_df = pd.read_csv(test_csv_path)
    all_data = pd.read_csv(train_csv_path)

    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((args.image_size, args.image_size)),
        transform.RandomHorizontalFlip(p=0),  # 师弟一旦加上病灶信息这里就不能使用数据增强,因为一旦数据增强就对应不上了
        transform.RandomVerticalFlip(p=0),
        transform.RandomResizedCrop(args.image_size),
        transform.ToTensor(),
    ])

    train_dataset = MultiviewImgDataset_mask(TRAIN_PATH, MASK_PATH, all_data, transform=transform_train,Single=False, no_mask=False, k=args.k)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,pin_memory=False)
    # 配置日志
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    logging.basicConfig(filename='{}/{}.log'.format(LOGS_DIR, formatted_now), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
        print(f"{key}: {value}")
    main(args, device)
    logging.shutdown()




