#Importing necessary libraries for computing and plotting
import numpy as np
import matplotlib.pyplot as plt

#Linear Regression Class
class linearRegression():
    def __init__(self):
        #LOADING training dataset from external file
        self.dataset_features = np.loadtxt('ex2x.dat')
        self.dataset_labels = np.loadtxt('ex2y.dat')

        #Splitting dataset into training and testing data
        self.xData = np.array(self.dataset_features)
        self.yData = np.array(self.dataset_labels)
        self.training_features , self.test_features = self.xData[:40], self.xData[40:]
        self.training_labels , self.test_labels = self.yData[:40], self.yData[40:]

    def trainModel(self, trainingData, trainingLabels):
        
    #Tester function to test variables and vectors
    def outt(self):
        print (self.training_features)
        print (self.training_labels)


model = linearRegression()
model.outt()
