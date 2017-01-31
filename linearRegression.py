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
        self.training_features = np.array(self.dataset_features)
        self.training_labels = np.array(self.dataset_labels)

    #function to calculate squared errors value
    def ComputeCost(self,X, y, theta):
        m = y.size
        predictions = theta[0] + X.dot(theta[1])
        sqErrors = (predictions - y)**2
        J = (1./(2.*m)) * sqErrors.sum()
        return J

    #Gradient Descent Fucntion to calculate best fit theta values
    def GradientDescent(self, X, y, theta, alpha, iters):
        jValues = np.zeros(shape=(iters))
        m = jValues.size
        for i in range(iters):
            J = self.ComputeCost(X,y, theta)
            temp0 = theta[0] - alpha*(1/m)* J
            temp1 = theta[1] - alpha*(1/m)* J
            theta[0] = temp0
            theta[1] = temp1
            #print (theta)
            jValues[i] = self.ComputeCost(X,y,theta)
        return theta, jValues

    #Tester function to test variables and vectors
    def outt(self):
        #print (self.training_features)
        #print (self.training_labels)
        iters = 1000
        alpha = 0.01
        theta = np.zeros(2)
        thetaFinal, J = self.GradientDescent(self.training_features, self.training_labels, theta, alpha, iters)
        print ("Final Theta Values: ", thetaFinal)
        print (J[:5])
        plt.xlabel("Age(Years)")
        plt.ylabel("Height(Meters)")
        plt.scatter(self.training_features, self.training_labels)
        plt.plot(self.training_features, self.training_labels)
        plt.show()
        x = np.zeros(J.size)
        for i in range(0,x.size-1):
            x[i] = i
        plt.scatter(x, J)
        plt.show()


model = linearRegression()
model.outt()
