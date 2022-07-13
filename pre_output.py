import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import DENSE, Normalize, Compose
from dataset.transforms import ToTensor
from model import Var_Nine
import numpy as np


def get_data_loaders(data_dir, batch_size=None, num_workers=None, k_dis=False):
    # normalize = Normalize(mean=DENSE.mean(), std=DENSE.std())
    transforms = Compose([
        ToTensor()
    ])

    test_loader = DataLoader(DENSE(root=data_dir, split='test', transform=transforms, k_dis=k_dis),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_loader


def run(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = Var_Nine(num_classes, args.k_dis)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    test_loader = get_data_loaders(args.dataset_dir, args.val_batch_size, args.num_workers, args.k_dis)
    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    model.eval()
    for data in tqdm(test_loader):
        for i, _ in enumerate(data[0]):
            data[0][i] = list(data[0][i].permute(1, 0, 2, 3))
        n = len(data[0][0])
        size = data[0][0][0].numel()
        file_id = data[2]
        start = data[-1]
        pred_labels = []
        with torch.no_grad():
            # Get predictions
            for i in np.arange(len(data[0][0])):
                distance = data[0][0][i].cuda()
                intensity = data[0][1][i].cuda()
                if args.k_dis:
                    k_dis = data[0][2][i].cuda()
                    params_list = [distance.unsqueeze(1), intensity.unsqueeze(1), k_dis.unsqueeze(1)]
                else:
                    params_list = [distance.unsqueeze(1), intensity.unsqueeze(1)]
                pred = model(params_list)
                pred = torch.squeeze(torch.argmax(pred, dim=1))
                pred_label = pred.transpose(1, 0).reshape(-1)
                pred_labels.append(pred_label)
        pred_labels = torch.stack(pred_labels).reshape(-1)
        target_labels = data[1].transpose(-1, -2).reshape(-1)
        if start > 0:
            pred_output = torch.cat((pred_labels[:(n - 1) * size], pred_labels[start:])).cpu().numpy()
            target = torch.cat((target_labels[:(n - 1) * size], target_labels[start:])).cpu().numpy()
        else:
            pred_output = pred_labels.cpu().numpy()
            target = target_labels.cpu().numpy()
        TP = np.sum((pred_output >= 2) & (target >= 2))
        FN = np.sum((pred_output < 2) & (target >= 2))
        FP = np.sum((pred_output >= 2) & (target < 2))
        TP_sum += TP
        FN_sum += FN
        FP_sum += FP
        # print(TP / (TP + FN))
        result_label_file = os.path.join(args.output_dir, ''.join(file_id) + '_pred.label')
        pred_output.astype('int32').tofile(result_label_file)
    print('Test Noise IOU Value according to your input model is: %.3f' % (TP_sum / (TP_sum + FN_sum + FP_sum)))


if __name__ == '__main__':
    parser = ArgumentParser('WeatherNet with PyTorch')
    parser.add_argument('--resume', type=str,
                        default='checkpoints/dense_basicconv_ks=1_nodrop_nolila/dense_basicconv_ks=1_nodrop_nolila_epoch243_mIoU=93.9.pth',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='input batch size for validation to visualize result, Must Be 1')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--output-dir', default='result/93.9',
                        help='directory to save model checkpoints')
    parser.add_argument("--dataset-dir", type=str, default="/data1/mayq/datasets/cnn_denoising",
                        help="location of the dataset")
    parser.add_argument('--k_dis', type=bool, default=False,
                        help='add k-means distance channel')

    run(parser.parse_args())
