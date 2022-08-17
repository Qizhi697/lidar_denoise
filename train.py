import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU, mIoU, Metric
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
from collections import OrderedDict
from dataset import DENSE, Normalize, Compose, RandomHorizontalFlip
from dataset.transforms import ToTensor
from model import Var_Nine
from ignite.utils import manual_seed


def get_data_loaders(data_dir, batch_size, val_batch_size, num_workers, k_dis):
    # normalize = Normalize(mean=DENSE.mean(), std=DENSE.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])

    val_transforms = Compose([
        ToTensor()
    ])

    train_loader = DataLoader(DENSE(root=data_dir, split='train', transform=transforms, k_dis=k_dis),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(DENSE(root=data_dir, split='test', transform=val_transforms, k_dis=k_dis),
                            batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def save(obj, dir, file_name):
    save_file = os.path.join(dir, file_name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    torch.save(obj, save_file)


class PR(Metric):
    def __init__(self, H, W):
        super(PR, self).__init__()
        self.H = H
        self.W = W

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, output):
        """
        0 : no_label points
        1 : clear points
        2, etc : noisy points
        """
        pred_value, target = output[0].detach(), output[1].detach()
        num_classes = pred_value.shape[1]
        pred = torch.argmax(pred_value, dim=1)
        pred_noise = torch.where((pred < num_classes) & (pred >= 2), 1, 0)
        gt_noise = torch.where((target < num_classes) & (target >= 2), 1, 0)
        pred_not_noise = torch.where(pred < 2, 1, 0)
        gt_not_noise = torch.where(target < 2, 1, 0)
        self.TP += torch.sum(pred_noise & gt_noise)
        self.FP += torch.sum(pred_noise & gt_not_noise)
        self.FN += torch.sum(pred_not_noise & gt_noise)

    def compute(self):
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        return torch.tensor([precision, recall])


class PWCELoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'none') -> None:
        super(PWCELoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target, point_weight):
        loss_none = super(PWCELoss, self).forward(input, target)
        assert point_weight.shape == loss_none.shape
        return (loss_none * point_weight).sum() / (self.weight[target] * point_weight).sum()


def run(args):
    if args.seed is not None:
        manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = Var_Nine(num_classes, args.k_dis)
    model = model.to(device)
    train_loader, val_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.val_batch_size,
                                                args.num_workers, args.k_dis)
    CEloss = nn.CrossEntropyLoss().to(device)
    criterion_dense = nn.CrossEntropyLoss(weight=DENSE.class_weights()).to(device)
    # criterion_none = nn.CrossEntropyLoss(weight=DENSE.class_weights(), reduction='none').to(device)
    # criterion = PWCELoss(weight=DENSE.class_weights()).to(device)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # new_state_dict = OrderedDict()
            # for k, v in checkpoint.items():
            #     name = k[7:]  # remove `module.`
            #     new_state_dict[name] = v
            # model.load_state_dict(new_state_dict)
            model.load_state_dict(checkpoint)
            # args.start_epoch = checkpoint['epoch']
            # model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("Loaded checkpoint '{}' (Epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    def _prepare_batch(batch, non_blocking=True):
        param_lists, target = batch[:2]
        if args.k_dis:
            return ([param_lists[0].cuda(), param_lists[1].cuda(), param_lists[2].cuda()], target.cuda())
        else:
            return ([param_lists[0].cuda(), param_lists[1].cuda()], target.cuda())

    def _update(engine, batch):
        model.train()

        if engine.state.iteration % args.grad_accum == 0:
            optimizer.zero_grad()
        param_lists, target = _prepare_batch(batch)
        pred = model(param_lists)
        # Td = torch.amax(param_lists[0], dim=(2, 3)) / 2
        # Td = 100
        # Ti = 0.3
        # weight = torch.squeeze(torch.abs((Td * Ti) / (distance * reflectivity)))
        # weight = torch.squeeze(torch.abs(Td[:, :, None, None] / param_lists[0]), dim=1)
        # mask = torch.ones_like(target)
        # mask[(target != 2) | (weight < 1)] = 0
        # mask[(target == 0) | (target == 1) | (weight < 1)] = 0
        # mask[((target != 2) & (weight > 1)) | ((target == 2) & (weight < 1))] = 0
        # weight[mask == 0] = 1
        # weight[weight > 10] = 10
        # class_weight = DENSE.class_weights()[target].cuda()
        # loss = (criterion_none(pred, target) * weight).sum() / (args.grad_accum * (class_weight * weight).sum())
        # loss = criterion(pred, target) / args.grad_accum
        # loss = criterion(pred, target, weight)
        loss = criterion_dense(pred, target)
        engine.state.metrics['loss'] = loss
        loss.backward()
        if engine.state.iteration % args.grad_accum == 0:
            optimizer.step()
        # train_fraction_done = engine.state.iteration / train_num_batch
        # writer.add_scalar('Loss/train', loss.item(),
        #                   ((engine.state.epoch-1) + train_fraction_done) * train_num_batch * args.batch_size)
        return loss.item()

    trainer = Engine(_update)

    # attach running average metrics
    # RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    # attach progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss'])

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            param_lists, target = _prepare_batch(batch)
            pred = model(param_lists)
            # Td = torch.amax(param_lists[0], dim=(2, 3)) / 2
            # pweight = torch.squeeze(torch.abs(Td[:, :, None, None] / param_lists[0]), dim=1)
            # mask = torch.ones_like(target)
            # mask[(target != 2) | (pweight < 1)] = 0
            # mask[(target == 0) | (target == 1) | (pweight < 1)] = 0
            # pweight[mask == 0] = 1
            # pweight[pweight > 10] = 10

            # return pred, target, pweight
            return pred, target

    evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes)
    IoU(cm, ignore_index=0).attach(evaluator, 'IoU')
    Loss(criterion_dense).attach(evaluator, 'loss')
    # Loss(criterion, output_transform=lambda x: (x[0], x[1], dict(point_weight=x[2]))).attach(evaluator, 'loss')
    mIoU(cm, ignore_index=0).attach(evaluator, 'mIoU')
    PR(64, 50).attach(evaluator, 'PR')
    Loss(CEloss, output_transform=lambda x: (x[0], x[1])).attach(evaluator, 'CELoss')

    pbar2 = ProgressBar(persist=True, desc='Eval Epoch')
    pbar2.attach(evaluator)

    def _global_step_transform(engine, event_name):
        if trainer.state is not None:
            return trainer.state.iteration
        else:
            return 1

    tb_logger = TensorboardLogger(args.log_dir)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag='training',
                                               metric_names=['loss']),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation',
                                               metric_names=['loss', 'CELoss', 'IoU', 'mIoU', 'PR'],
                                               global_step_transform=_global_step_transform),
                     event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        if args.resume:
            engine.state.epoch = args.start_epoch

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        epoch = trainer.state.epoch if trainer.state is not None else 1
        iou = engine.state.metrics['IoU'] * 100.0
        # mean_iou = iou.mean()
        # name = 'epoch{}_mIoU={:.1f}.pth'.format(epoch, mean_iou)
        name = 'epoch{}_mIoU={:.1f}.pth'.format(epoch, iou[-1])
        # file = {'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'args': args}
        # save(file, os.path.join('checkpoints', args.output_dir), 'checkpoint_var6_{}'.format(name))
        save(model.state_dict(), os.path.join('checkpoints', args.output_dir), '{}_{}'.format(args.output_dir, name))

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        pbar.log_message("Start Validation - Epoch: [{}/{}]".format(engine.state.epoch, engine.state.max_epochs))
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        iou = metrics['IoU'] * 100.0
        pr = metrics['PR']
        mean_iou = iou.mean()

        iou_text = ', '.join(['{}: {:.1f}'.format(DENSE.classes[i + 1].name, v) for i, v in enumerate(iou.tolist())])
        pr_text = ', '.join(['{}: {:.4f}'.format(DENSE.pr[i].name, v) for i, v in enumerate(pr.tolist())])
        pbar.log_message("Validation results - Epoch: [{}/{}]: Loss: {:.2e}\n IoU: {}\n mIoU: {:.1f}\n PR: {}"
                         .format(engine.state.epoch, engine.state.max_epochs, loss, iou_text, mean_iou, pr_text))

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")

            name = 'epoch{}_exception.pth'.format(trainer.state.epoch)
            file = {'model': model.state_dict(), 'epoch': trainer.state.epoch, 'optimizer': optimizer.state_dict(),
                    'args': args}

            save(file, os.path.join('checkpoints', args.output_dir), 'checkpoint_model_{}'.format(name))
            save(model.state_dict(), os.path.join('checkpoints', args.output_dir), 'model_{}'.format(name))
        else:
            raise e

    if args.eval_on_start:
        print("Start validation")
        evaluator.run(val_loader, max_epochs=1)

    print("Start training")
    trainer.run(train_loader, max_epochs=args.epochs)
    tb_logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('WeatherNet with PyTorch')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=16,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=600,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=8e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='manual seed')
    parser.add_argument('--output-dir', default='sim_34_110')
    parser.add_argument('--resume', type=str,
                        default='',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--dataset-dir", type=str, default="/data1/mayq/datasets/WADS",
                        help="location of the dataset")
    parser.add_argument("--eval-on-start", type=bool, default=False,
                        help="evaluate before training")
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='grad accumulation')
    parser.add_argument('--k_dis', type=bool, default=True,
                        help='add k-means distance channel')

    run(parser.parse_args())
