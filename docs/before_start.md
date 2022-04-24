# Getting Started
we visualize our training details via wandb (https://wandb.ai/site). 
### visualization
1) you'll need to login
   ```shell 
   $ wandb login
   ```
2) you'll need to copy & paste you API key in terminal
   ```shell
   $ https://wandb.ai/authorize
   ```
   or paste it to the **Code/VnetLA/Configs/config.py**
   ```shell
   os.environ['WANDB_API_KEY'] = "you key"
   ```
### training
run the scripts with your prefered labelled number

```shell
./scripts/train_la.sh 8
```

### inference
fullfill your environment name below
```shell
python3 Code/VnetLA/validate.py --env_name ""
```

## Setting
We offer different training iterations upon the labelled num.

**1). LA dataset**

   | # labels |  max_iterations | unsup weight |
   |----------------|-----------------|--------------|
   | 8              | 9000            | 0.3          |
   | 16             | 9000            | 1.0          |
   | 32             | 12000           | 1.0          |

**2). BRaTS19 dataset**

| # labels |  max_iterations | unsup weight |
|----------------|-----------------|--------------|
| 25              | 30000            | 1.0          |
| 50              | 50000            | 1.0          |
| 125             | 60000           | 1.0          |
