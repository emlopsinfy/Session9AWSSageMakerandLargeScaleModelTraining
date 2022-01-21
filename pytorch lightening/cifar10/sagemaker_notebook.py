import sagemaker
import uuid

sagemaker_session = sagemaker.Session()
print('sagemaker version:' + sagemaker.__version__)
--
role = sagemaker.get_execution_role()
role
---
bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/emlo-s9-pt-mnist'

checkpoint_suffix = str(uuid.uuid4())[:8]
checkpoint_s3_path = 's3://{}/checkpoint-{}'.format(bucket, checkpoint_suffix)
#checkpoint_s3_path = 's3://sagemaker-ap-south-1-426011120934/checkpoint-b56bdf03'

print(f'checkpointing path : {checkpoint_s3_path}')
--
import os
import subprocess 

instance_type = 'local'

if subprocess.call('nvidia-smi') == 0:
    # Set type to GPU if one is present
    instance_type = 'local_gpu'
    
print('instance_type:', instance_type)
---
pip install pytorch-lightning --quiet
---
from cifar10_pl import CIFAR10DataModule
dm = CIFAR10DataModule(batch_size=32)
dm.prepare_data()
dm.setup()
trainloader = dm.train_dataloader()
testloader = dm.test_dataloader()

# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
	--
import numpy as np
import torchvision, torch
import matplotlib.pyplot as plt

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
--
inputs = sagemaker_session.upload_data(path = "data", bucket = bucket, key_prefix="data/cifar10")
--
from sagemaker.pytorch import PyTorch
--
use_spot_instances = True
max_run = 800
max_wait = 1200 if use_spot_instances else None
--
from sagemaker.pytorch import PyTorch

hyperparameters = {'batch_size': 32, 'checkpoint-path':checkpoint_s3_path}

cifar10_estimator = PyTorch(
    entry_point = "cifar10_pl.py",
    role=role,
    framework_version="1.7.1",
    py_version="py3",
    hyperparameters=hyperparameters,
    instance_count=2,
    instance_type="ml.g4dn.4xlarge",
    base_job_name = 'cifar10-pytorch-Jan18-v1-2022-spot',
    checkpoints_s3_uri = checkpoint_s3_path,
    debugger_hook_config = False,
    use_spot_instances = use_spot_instances,
    max_run = max_run,
    max_wait = max_wait
)
--
cifar10_estimator.fit(inputs)
--
from sagemaker.pytorch import PyTorchModel

predictor = cifar10_estimator.deploy(initial_instance_count=1, instance_type="ml.c4.8xlarge")
---
# get some test images

dataiter = iter(testloader)
images, labels = dataiter.next()
print(images.size())

# print images
imshow(torchvision.utils.make_grid(images))
print('Ground Truth :', ' '.join('%4s' % classes[labels[j]] for j in range(4)))

outputs = predictor.predict(images.numpy())

_, predicted = torch.max(torch.from_numpy(np.array(outputs)), 1)

print('Predicted : ', ' '.join('%4s' % classes[predicted[j]] for j in range(4)))
--

predictor.delete_endpoint()  #Very Important !!!

--