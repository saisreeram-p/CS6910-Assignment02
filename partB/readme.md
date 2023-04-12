# CS6910 Assignment 2
## Author : P.Sai Sree Ram CS22M076
### Instructions to create a model , train , predict CNN Network :
1. Install the required modules/libraries before running
2. Maintain proper folder structure of the dataset
```
baseDir = inaturalist_12K
trainDir = baseDir+"/train/"
testDir = baseDir+"/val/"
```
3. To To create a model
```
Resnet1 = Resnet()

```
4. To train the modal
```
trainer = pl.Trainer(max_epochs=5)
trainer.fit(Resnet1,train_dataset,val_dataset)
```
5.To Predict on val dataset given for testing
```
trainer.test(Resnet1, test_dataset)

```
