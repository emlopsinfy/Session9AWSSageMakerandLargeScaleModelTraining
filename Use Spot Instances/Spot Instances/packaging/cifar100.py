import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.classifiers import *
from datasets.CIFAR import *
from datasets.LSUN import *
from datasets.SVHN import *

import logging,argparse,json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',  'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',  'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',  'bottles', 'bowls', 'cans', 'cups', 'plates',  'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',  'clock', 'computer keyboard', 'lamp', 'telephone', 'television',  'bed', 'chair', 'couch', 'table', 'wardrobe',  'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',  'bear', 'leopard', 'lion', 'tiger', 'wolf',  'bridge', 'castle', 'house', 'road', 'skyscraper',  'cloud', 'forest', 'mountain', 'plain', 'sea',  'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',  'fox', 'porcupine', 'possum', 'raccoon', 'skunk',  'crab', 'lobster', 'snail', 'spider', 'worm',  'baby', 'boy', 'girl', 'man', 'woman',  'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',  'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',  'maple', 'oak', 'palm', 'pine', 'willow',  'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',  'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')


def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)
   
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        print(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device Type: {}".format(device))

    print("Loading Cifar10 dataset")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)

    print("Model loaded")
    model = Net()

    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Check if checkpoints exists
    if not os.path.isfile(args.checkpoint_path + '/checkpoint.pth'):
        epoch_number = 0
    else:    
        model, optimizer, epoch_number = _load_checkpoint(model, optimizer, args)        
    
    for epoch in range(epoch_number, args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
        _save_checkpoint(model, optimizer, epoch, loss, args)
            
    print('Finished Training')
    return _save_model(model, args.model_dir)


def _save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def _save_checkpoint(model, optimizer, epoch, loss, args):
    print("epoch: {} - loss: {}".format(epoch+1, loss))
    checkpointing_path = args.checkpoint_path + '/checkpoint.pth'
    print("Saving the Checkpoint: {}".format(checkpointing_path))
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, checkpointing_path)

    
def _load_checkpoint(model, optimizer, args):
    print("--------------------------------------------")
    print("Checkpoint file found!")
    print("Loading Checkpoint From: {}".format(args.checkpoint_path + '/checkpoint.pth'))
    checkpoint = torch.load(args.checkpoint_path + '/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_number = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Checkpoint File Loaded - epoch_number: {} - loss: {}".format(epoch_number, loss))
    print('Resuming training from epoch: {}'.format(epoch_number+1))
    print("--------------------------------------------")
    return model, optimizer, epoch_number

    
def model_fn(model_dir):
    print('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def _train_ligthning(args):
    
    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)
    
    
    dm = CIFAR100DataModule()
    
    model_name = 'CIFAR100'+'_'+'Resnet'
    model = globals()[model_name]()
    
    checkpoint_callback=ModelCheckpoint(filepath=args.checkpoint_path)
    trainer=Trainer(checkpoint_callback=checkpoint_callback, gpus=args.num_gpus, num_nodes=1, max_epochs = args.epochs)
    
    trainer.fit(model, dm)
    trainer.save_checkpoint(os.path.join(args.checkpoint_path ,'final.ckpt'))
    trainer.test(model, datamodule = dm)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
    
    
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train_ligthning(parser.parse_args())

