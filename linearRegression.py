#Importing necessary libraries for computing and plotting
import numpy as np
import matplotlib.pyplot as plt

#Linear Regression Class
class linearRegression():
    def __init__(self):
        #LOADING training dataset from external file
        self.dataset_features = np.loadtxt('ex2x.dat')
        self.dataset_labels = np.loadtxt('ex2y.dat')
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
    #This is the most important function
    def GradientDescent(self, X, y, theta, alpha, iters):
        jValues = np.zeros(shape=(iters))
        m = y.size
        for i in range(iters):
            predictions = theta[0] + X.dot(theta[1])
            err1 = (predictions - y)
            err2 = (predictions - y) * X
            theta[0] -= alpha * (1./m) * err1.sum()
            theta[1] -= alpha * (1./m) * err2.sum()
            jValues[i] = self.ComputeCost(X,y,theta)
        return theta, jValues

    #Function to run our program
    def run(self):
        #print (self.training_features)
        #print (self.training_labels)
        iters = 1500
        alpha = 0.07
        theta = np.zeros(2)
        thetaFinal, J = self.GradientDescent(self.training_features, self.training_labels, theta, alpha, iters)

        #Our final training parameters
        print ("Final Theta Values: ", thetaFinal)
        plt.xlabel("Age(Years)")
        plt.ylabel("Height(Meters)")
        plt.scatter(self.training_features, self.training_labels)
        plt.plot(self.training_features, theta[0] + self.training_features.dot(theta[1]))
        plt.show()


model = linearRegression()
model.run()
