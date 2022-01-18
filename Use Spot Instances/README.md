https://github.com/aws-samples/amazon-sagemaker-managed-spot-training/blob/main/pytorch_managed_spot_training_checkpointing/pytorch_managed_spot_training_checkpointing.ipynb



Every time, we run training on SageMaker, it creates training job for us.

from sagemaker.pytorch import PyTorch

hyperparameters = {‘epochs’:10}

cifar10_estimator = PyTorch(

​	entry_point = ‘source_dir/cifar10.py’,

​	role = role,

​	framework_version = “1.7.1”,

​	py_version = “py3”,

​	hyperparameters = hyperparameters,

​	instance_count = 1,

​	instance_type = “m1.p3.2xlarge”,

​	base_job_name = ‘cifar10-pytorch-saturday’

)

cifar10_estimator.fit(inputs)



##### Spot Instances

##### There will be 70% discount if we use spot instance, because AWS provides you the available machine, not the machine we want, so there is change in total seconds trained and total seconds billled.



use_spot_instances = True

max_run = 600

max_wait = 1200 if use_spot_instances else None



from sagemaker.pytorch import PyTorch

hyperparameters = {‘epochs’:5}

cifar10_estimator = PyTorch(

​	entry_point = ‘source_dir/cifar10.py’,

​	role = role,

​	framework_version = “1.7.1”,

​	py_version = “py3”,

​	hyperparameters = hyperparameters,

​	instance_count = 1,

​	instance_type = “m1.p3.2xlarge”,

​	base_job_name = ‘cifar10-pytorch-saturday-spot’,

​	checkpoints_s3_uri = checkpoint_s3_path,

​	debugger_hook_config= False,

​	use_spot_instance = use_spot_instance,

​	max_run = max_run,

​	max_wait = max_wait

)

cifar10_estimator.fit(inputs)

