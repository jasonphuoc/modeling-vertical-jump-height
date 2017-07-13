from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
    
#input training set

trainingSet = SupervisedDataSet(2, 1)
inputTrainSet = file('../test-data/vertical-jump-height-nfl.csv', 'r').readlines()[10:30]
for line in inputTrainSet:
    splitLine = line.split(',')
    trainingSet.addSample((splitLine[5], splitLine[2]), splitLine[6])

#input validation set

validationSet = SupervisedDataSet(2, 1)
inputValidationSet = file('../test-data/vertical-jump-height-nfl.csv', 'r').readlines()[30:44]
for line in inputValidationSet:
    splitLine = line.split(',')
    validationSet.addSample((splitLine[5], splitLine[2]), splitLine[6])

#build network

net = buildNetwork(2, 1, 1, bias=True)

#train network

trainer = BackpropTrainer(net, validationSet, learningrate = 0.001, momentum = 0.99)
trainer.trainUntilConvergence(verbose=True,
                              trainingData=trainingSet,
                              validationData=validationSet,
                              maxEpochs=100)

#test network

inputTestSet = file('../test-data/vertical-jump-height-nfl.csv', 'r').readlines()[1:10]
for line in inputTestSet:
    splitLine = line.strip().split(',')
    print splitLine[0], net.activate([splitLine[5], splitLine[2]]) 
