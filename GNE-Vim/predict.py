import logging
import os

import numpy as np
import torch
import torchvision
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets2 import IQADataset
from options.test_options import TestOptions
from utils.process_image import ToTensor, five_point_crop, Normalize
from utils.util import setup_seed, set_logging
from models.gne_vim import VimIQAModel, Discriminator, ScoreModel


class Test:
    def __init__(self, config):
        self.opt = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_model()
        self.init_data()

    def init_model(self):
        if not self.opt.baseline:
            self.model = VimIQAModel(checkpoint=self.opt.weight, model_type=self.opt.model_type)
        else:
            self.model = ScoreModel(checkpoint=self.opt.weight, model_type=self.opt.model_type)
        self.model.to(self.device)
        self.load_model()
        self.model.eval()

    def init_data(self):
        test_dataset = IQADataset(
            ref_path=self.opt.val_ref_path,
            dis_path=self.opt.val_dis_path,
            txt_file_name=self.opt.val_list,
            transform=torchvision.transforms.Compose([
                # torchvision.transforms.Resize(self.opt.crop_size),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          var=(0.229, 0.224, 0.225))
            ]),
        )
        logging.info('number of test scenes: {}'.format(len(test_dataset)))

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=False,
            shuffle=False,
        )

    def load_model(self):
        model_path = self.opt.ckpt
        if not os.path.exists(model_path):
            raise ValueError(f'Model not found at {model_path}')

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])  # )
        logging.info(f'Loaded model from {model_path}')

    def test(self):
        with torch.no_grad():
            names = []
            pred_epoch = []
            labels_epoch = []

            with tqdm(desc='Testing', unit='it', total=len(self.test_loader)) as pbar:
                for _, data in enumerate(self.test_loader):
                    pred = 0
                    for i in range(self.opt.num_avg_val):
                        d_img_org = data['d_img_org'].cuda()
                        r_img_org = data['r_img_org'].cuda()
                        d_img_org, r_img_org = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
                        score = self.model(d_img_org)
                        pred += score
                    pred /= self.opt.num_avg_val
                    labels = data['score']
                    labels = labels.type(torch.FloatTensor).cuda()

                    names.extend(data['d_img_name'])
                    # 保存结果
                    pred_batch_numpy = pred.data.cpu().numpy()
                    labels_batch_numpy = labels.data.cpu().numpy()
                    pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                    labels_epoch = np.append(labels_epoch, labels_batch_numpy)
                    pbar.update()

            # 计算总体相关系数
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            msg = f'\nOverall:\nSRCC: {rho_s:.4f}\nPLCC: {rho_p:.4f}'
            print(msg)
            logging.info(msg)

            # 保存预测结果为csv文件，（图像名称，预测分数，标签分数）
            save_path = os.path.join(self.opt.checkpoints_dir, f'{self.opt.dataset.lower()}_ours_results.csv')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                for name, pred, gt in zip(names, pred_epoch, labels_epoch):
                    f.write(f'{name},{pred},{gt}\n')
            logging.info(f'Saved results to {save_path}')
            print(f'Saved results to {save_path}')


if __name__ == '__main__':
    config = TestOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    tester = Test(config)
    tester.test()
