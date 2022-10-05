import datetime
import os
import time

import pickle
import copy
import shutil
from pathlib import Path
import tempfile
from collections import Counter

from typing import Iterable
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image
import torchmetrics

import utils

import mlflow
import mlflow.pytorch
import optuna

from factories import OptimizerFactory, DatasetFactory, ModelFactory, LossFactory, LRSchedulerFactory, TransformFactory
mlflow.set_tracking_uri('http://mlflow.172.26.62.216.nip.io')

        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def calc_log_ROC(model, data_loader, device, classes):
    #calculate roc curve, create and save matplotlib fig, send to mlflow
    
    # only tested for binary classification, not distributed mode and gpu
    model.eval()
    with torch.no_grad():
        result = torch.cuda.FloatTensor()
        targets = torch.cuda.LongTensor()
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            m = nn.Softmax(dim=1)
            output =  m(output)
            result = torch.cat((result,output),dim=0)
            targets = torch.cat((targets,target),dim=0)
            
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    class_of_interst = len(classes)-1

    y = targets.cpu()
    scores = result[:,class_of_interst].cpu()
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=class_of_interst)
    roc_auc = auc(fpr, tpr)
            
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        savepath = os.path.join(tmpdirname,'ROC.png')
        plt.savefig(savepath, dpi=1000)
        mlflow.log_artifact(savepath, artifact_path="ROC_Curve")
        
def log_dataset(dataset, desc):
    classes = dataset.classes
    imgs = dataset.imgs
    labels = dataset.labels
    df = pd.DataFrame(list(zip(imgs, labels)),
                      columns =['img_path',classes[-1]])

    with tempfile.TemporaryDirectory() as tmpdirname:
        df.to_csv(tmpdirname+desc, index=False)
        
        mlflow.log_artifacts(tmpdirname, artifact_path="data")
    
def log_datasets(dataset, indices, splitpoint):
    classes = dataset.classes
    imgs = dataset.imgs
    labels = dataset.labels
    df = pd.DataFrame(list(zip(imgs, labels)),
                      columns =['img_path',classes[-1]])
    df_train = df.iloc[indices[:splitpoint]]
    df_test = df.iloc[indices[splitpoint:]]
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        df_train.to_csv(tmpdirname+"/train_set.csv", index=False)
        df_test.to_csv(tmpdirname+"/test_set.csv", index=False)
        
        mlflow.log_artifacts(tmpdirname, artifact_path="data")

def reshape_classification_head(model, model_architecture, num_classes):
    if model_architecture in ['resnet18',
                              'resnet34',
                              'resnet50',
                              'resnet101',
                              'resnet152',
                              'resnext50_32x4d',
                              'resnext101_32x8d',
                              'wide_resnet50_2',
                              'wide_resnet101_2',
                              'shufflenet_v2_x0_5',
                              'shufflenet_v2_x1_0']:
        num_ftrs = model.fc.in_features
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_architecture in ['squeezenet1_1']:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif model_architecture in ['densenet121',
                                'densenet169',
                                'densenet201',
                                'densenet161']:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f'changing classification head for model "{model_architecture}" not supported, exiting...')
    return model
        
    
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torchmetrics.functional.accuracy(output, target)
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['accuracy'].update(acc.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        
def calc_metrics(preds, target, class_names, metrics, per_class_metrics) -> dict():
    num_classes = len(class_names)
    metric_dict = {}
    # add single value metrics
    for metric in metrics:
        metric_dict[metric] = torchmetrics.functional.__dict__[metric](preds, target)
    
    # add multi value metrics
    for metric in per_class_metrics:
        metric_value = torchmetrics.functional.__dict__[metric](preds, target, average = None, num_classes = num_classes)
        for i,name in enumerate(class_names):
            metric_dict[metric+' '+name] = metric_value[i]
    return metric_dict

def evaluate(model, criterion, data_loader, device, class_names, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            
            metrics = calc_metrics(output,target,class_names,['accuracy'],['recall','precision'])
            for k,v in metrics.items():
                metric_logger.meters[k].update(v.item(), n=batch_size)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Acc {top1.global_avg:.3f}'
          .format(top1=metric_logger.accuracy))
        
    all_metrics = {k: v.global_avg for k, v in metric_logger.meters.items()}
    
    return all_metrics



def load_data(args):
    # Data loading code
    print("Loading data")
    print("Loading training data")
    st = time.time()

    dataset_factory = DatasetFactory()
    transform_factory = TransformFactory()
    dataset_train = dataset_factory.create_dataset('CSVDataset',
                                                   args.imgs_path,
                                                   args.data_path if args.data_path else args.data_paths[0],
                                                   transforms.Compose([transform_factory.create_transform('RandomResizedCrop', args),
                                                                       transform_factory.create_transform('RandomHorizontalFlip', args),
                                                                       transform_factory.create_transform('ToTensor', args),
                                                                       transform_factory.create_transform('Normalize', args)]))
    
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = dataset_factory.create_dataset('CSVDataset',
                                                   args.imgs_path,
                                                   args.data_path if args.data_path else args.data_paths[1],
                                                   transforms.Compose([transform_factory.create_transform('Resize', args),
                                                                       transform_factory.create_transform('ToTensor', args),
                                                                       transform_factory.create_transform('Normalize', args)]))
    

    class_names = dataset_train.classes
    
    if args.data_path:
        splitpoint=int(len(dataset_train)*(1-args.test_split))
        indices = torch.randperm(len(dataset_train)).tolist()
        
        log_datasets(dataset_train, indices, splitpoint)

        dataset_train = torch.utils.data.Subset(dataset_train, indices[:splitpoint])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[splitpoint:])
    else:
        log_dataset(dataset_train, "/train_set.csv")
        log_dataset(dataset_test, "/test_set.csv")
    
    # simple oversampling of minority class ## did not improve at all
    # get number of samples for each class in dataset and calculate sampling ratio (for WeightedRandomSampler)
    #train_ys = [dataset.targets[i] for i in dataset_train.indices]
    #samples = dict(Counter(train_ys)) # alternativeley: Counter(i.item() for i in train_ys)
    #class_sample_count = [samples[0], samples[1]]
    #weights = 1 / torch.Tensor(class_sample_count)
    
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        # oversample minority class, so that model sees roughly same num_samples from each class
        #train_sampler = torch.utils.data.WeightedRandomSampler(weights[train_ys], (max(class_sample_count)*2))
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    return dataset_train, dataset_test, train_sampler, test_sampler, class_names


def main(args, trial=None):
    
    if args.deterministic:
        torch.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if utils.is_main_process():
        if args.output_dir:
            if Path(args.output_dir).exists():
                shutil.rmtree(args.output_dir)
            utils.mkdir(args.output_dir)

#    utils.init_distributed_mode(args)
    print(args)

    dataset, dataset_test, train_sampler, test_sampler, class_names = load_data(args)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model_factory = ModelFactory()
    model = model_factory.create_model(args.model, args.pretrained)

    model = reshape_classification_head(model, args.model, len(class_names))

    device = torch.device(args.device)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    loss_factory = LossFactory()
    criterion = loss_factory.create_loss('CrossEntropyLoss')

    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create_optimizer(args.optimizer, model.parameters(), args)

    lr_scheduler_factory = LRSchedulerFactory()
    lr_scheduler = lr_scheduler_factory.create_scheduler('StepLR', optimizer, args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return
    
    print("Start training")
    start_time = time.time()
    best_model_performance = 0.0
    best_model_epoch = 0
    
    if utils.is_main_process():
        # Log parameters
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
    
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_start_time = time.time()
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        train_stop_time = time.time() - train_start_time
        mlflow.log_metric('train_time_per_epoch',train_stop_time,step=epoch)
        
        lr_scheduler.step()
        
        val_start_time = time.time()
        all_metrics = evaluate(model, criterion, data_loader_test, device=device, class_names=class_names)
        val_stop_time = time.time() - val_start_time
        mlflow.log_metric('val_time_per_epoch',val_stop_time,step=epoch)
                
        if utils.is_main_process():
            # log metrics            
            for k,v in all_metrics.items():
                mlflow.log_metric(k,v,step=epoch)
                
            # prune run (trial) if its not going well
            if trial is not None:
                intermediate_value = all_metrics['accuracy']
                trial.report(intermediate_value, epoch)

                if trial.should_prune():
                    mlflow.set_tag('stopped by', 'pruning')
                    raise optuna.TrialPruned()

            #track best model
            if all_metrics['accuracy'] > best_model_performance:
                best_model = copy.deepcopy(model_without_ddp)
                best_model_optimizer = copy.deepcopy(optimizer)
                best_model_lr_scheduler = copy.deepcopy(lr_scheduler)
                best_model_performance = all_metrics['accuracy']
                best_model_epoch = epoch
                
            early_stopping(all_metrics['loss'])
            if early_stopping.early_stop:
                print("Early stopping")
                mlflow.set_tag('stopped by', 'early stopping')
                break
    else:
        mlflow.set_tag('stopped by', 'end of epochs')
    #save best model locally
    if args.output_dir:
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_model_optimizer.state_dict(),
            'lr_scheduler': best_model_lr_scheduler.state_dict(),
            'args': args}
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model.pth'))

    # log model
    if utils.is_main_process():
        if args.log_model:
            print("\nLogging the trained model as a run artifact...")
            mlflow.pytorch.log_model(best_model, artifact_path="pytorch-model", pickle_module=pickle)
            print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model"))
            
        # also save best model performance in separate metric
        mlflow.log_metric('best accuracy', best_model_performance, best_model_epoch)
            
#        calc_log_ROC(best_model, data_loader_test, device=args.device, classes=class_names)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best_model_performance


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    subparsers = parser.add_subparsers(help='sub-command help')
    
    # data
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', help='path to csv containing dataset')
    group.add_argument('--data-paths', nargs=2, help='path to train- and test-csv containing labels and paths to images')
    parser.add_argument('--imgs-path', default='data_small', help='folder containing all the images')
    parser.add_argument('--name', default='default', help='name of the training run')
    parser.add_argument('--experiment-name', default='', help='name of the experiment')
    parser.add_argument('--input-size', type=int, nargs=2, default=[150,150], help='shape to which images are resized for training: (h, w)')
    parser.add_argument('--test-split', type=float, default=0.2, help='portion of data to use for testing')
    
    # model
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16), rule of thumb: 4-8 workers per GPU')
    # optimizer
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use: SGD, Adam, RMSprop')
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    ### rmsprop
    parser.add_argument('--alpha', default=0.99, type=float, help='smoothing constant')
    ### adam
    parser.add_argument('--eps', default=1e-08, type=float, help='smoothing constant')
    
    # lr scheduler params
    parser.add_argument('--lr-step-size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        #action="store_true",
        default=True
    )
    parser.add_argument(
        "--log_model",
        dest="log_model",
        help="log model to mlflow",
        action="store_true"
    )
    parser.add_argument('--deterministic', action='store_true', help='turns on torch.backends.cudnn.deterministic and turns off torch.backends.cudnn.benchmark')

    # distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.name):
        main(args)
    #main(args)
