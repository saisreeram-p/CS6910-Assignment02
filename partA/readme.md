# CS6910 Assignment 2
## Author : P.Sai Sree Ram CS22M076
### Instructions to create a model , train , predict CNN Network :
1. Install the required modules/libraries before running

2. To run the train.py
 ```
python train.py --wandb_entity myname --wandb_project myprojectname

python train.py -dp inaturalist_12K -e 5 -b 16 -lr 0.00001 -a GELU -bn True -da False -do 0.3 -iz 256 -ks 3 -pks 3 -fz 64 -fl same
 ```
All the argumets are listed below 

| Name | Default Value | Description |
| :---: | :----------: | :--------|
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-dp`, `--dataset_path` | inaturalist_12K | Give folder name only no / |
| `-e`, `--epochs` | 1 | Number of epochs to train the model. |
| `-b`, `--batch_size` | 16 | Batch size used to train the model. |
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters |
| `-a`, `--activation`| GELU | choices: [ReLU, GELU, SiLU, Mish] |
| `-bn`, `--batch_normalisation` | False | To add batch_normalisation choices = [True , False] |
| `-da`, `--data_augmentation` | False | To add data_augmentation choices = [True , False] |
| `-do`, `--dropout` | False | Value of dropout |
| `-iz`, `--input_size` | 128 | Shape of the image while loading|
| `-ks`, `--kernel_size` | 3 | Conv kernel size |
| `-pks`, `--pool_kernel_size` | 2 | Max2dpool kernel size |
| `-fz`, `--filter_size` | 64 | filter size |
| `-fl`, `--filter_organisation` | same | choices = [same, half, double] filter_organisation with same size or half or double |

