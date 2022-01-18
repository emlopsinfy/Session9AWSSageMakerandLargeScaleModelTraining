# **Note on Distributed Training **

 

It is important to understand that the optimization in a distributed setting does not change at all when compared to the single-device setting, that is, we still minimize the same cost function with the same model and the same optimizer.
The real difference is that the gradient computation gets split into multiple devices and runs in parallel. This works simply because of the linearity of the gradient operator, i.e., computing the gradient for individual data samples and then averaging them is the same as computing the gradient using the whole batch of data at once.

 

![img](https://miro.medium.com/max/700/1*sKOrEP7PJK9sqNAnyYa2pw.png)

 

In summary, there are four main steps involved in a single training step

STEP1: We start off with the same model weights on all devices. Each device gets its own split of the data batch and performs a forward pass. This yields a different loss value per device.

![img](https://miro.medium.com/max/700/1*L96fdQ6_zNTIqZKU4gT0mQ.jpeg)

STEP2: Given the loss value, we can perform the backward pass which computes the gradients of the loss w.r.t. the model weights. We now have a different gradient per GPU device.

![img](https://miro.medium.com/max/700/1*GeXTEfUoPw4dZeirpo361w.jpeg)

STEP3: We synchronize the gradients by summing them up and dividing by the number of GPU devices involved. At the end of this process, each GPU now has the same averaged gradients.

 

STEP4:  Finally, all models can update their weights with the synchronized gradient. Because the gradient is the same on all GPUs, we again end up with the same model weights on all devices and the next training step can begin. 

Leveraging multiple GPUs in vanilla PyTorch can be [overwhelming (Links to an external site.)](https://github.com/pytorch/examples/blob/01539f9eada34aef67ae7b3d674f30634c35f468/imagenet/main.py), and to implement steps 1–4 from the theory above, a significant amount of code changes are required to “refactor” the codebase. With PyTorch Lightning, single node training with multiple GPUs is as trivial as adding two arguments to the Trainer:

![img](https://miro.medium.com/max/700/1*rDzoCVvqjT41Ax-Lm2F9TQ.png)

 

DPP stands for Distributed Data-Parallel. It is useful when you need to speed up training because you have a large amount of data, or working with a large batch size that cannot be fit into the memory of a single GPU.

![img](https://miro.medium.com/max/700/1*Z84Y5GZY4DV1nCC7SRi3jQ.png)

 

Pytorch Lightning provides [several (Links to an external site.)](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#distributed-modes) distributed strategies:

- Data Parallel (`strategy='dp'`) (multiple-gpus, 1 machine)
- DistributedDataParallel (`strategy='ddp'`) (multiple-gpus across many machines (python script based)).
- DistributedDataParallel (`strategy='ddp_spawn'`) (multiple-gpus across many machines (spawn based)).
- DistributedDataParallel 2 (`strategy='ddp2'`) (DP in a machine, DDP across machines).
- Horovod (`strategy='horovod'`) (multi-machine, multi-gpu, configured at runtime)
- TPUs (`tpu_cores=8|x`) (tpu or TPU pod)

 

[**SHARED MODEL TRAINING** (Links to an external site.)](https://medium.com/pytorch/pytorch-lightning-1-1-model-parallelism-training-and-more-logging-options-7d1e47db7b0b)

PL introduced SMT plugin that utilizes Data-Parallel training under the hood, but optimizer states and gradients are shared across GPUs. This means that the memory overhead per GPU is lower, as each GPU only has to maintain a partition of optimizer state and gradients. Using plugin can reduce memory requirements by upto 60% (NLP models).

 

Let's revise some of our PT understanding via [code (Links to an external site.)](https://colab.research.google.com/github/PytorchLightning/lightning-tutorials/blob/publication/.notebooks/lightning_examples/cifar10-baseline.ipynb).