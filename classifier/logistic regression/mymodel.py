
import numpy as np

def sigmoid(x):

        return 1 / (1 + np.exp(-x))
def tanh(x):
        return np.tanh(x) 
def log_loss(y, y_predict):
    return - np.mean(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
def accuracy(y,y_predict):
    return np.mean( y==y_predict )      
class LogisticRegression :
    
    def __init__(self,proba='sigmoid',cost='log_loss',learning_rate=0.1,solver='gd',max_iter=100,random_state=0,treshold_loss = 1e-5) -> None:
        self.coef=None
        self.proba=proba
        self.hypo = None
        self.setHypo()
        self.cost=log_loss
        self.accuracy = None
        self.learning_rate=learning_rate
        self.treshold=treshold_loss
        self.solver=solver
        self.max_iter=max_iter
        self.random_state=random_state
        np.random.seed(self.random_state)
    
    def fit(self,input,output) :
        x=np.array(input)
        shape=x.shape 
        # initialize coeff (+1 is for the bias term)
        self.coef=np.random.randn(shape[1]+1)
        
        #add the bias term
        x=np.hstack((np.ones((shape[0],1)),x))
        
        y=np.array(output)
        # get the Y_predict
        if self.solver=='gd':
            self.fitByGd(x,y)
        elif self.solver=='sgd':
            print('solving with sgd')
        elif self.solver=='mbsgd':
            print('solving with mbsgd')
    def fitByGd(self,x,y):
        #basic gradient descent
        for  i in range(self.max_iter):
            np.random.shuffle(x) 
            y_predict= x @ self.coef
            # proba of Y
            y_proba=self.hypo(y_predict)
            #calculate the error (log_loss)
            j=self.cost(y,y_proba)
            if(j < self.treshold):
                break
           
            #calculate the gradient
            gradient = (1 / len(y)) * x.T @ (y_proba - y)
            #update the coefficients
            self.coef=self.coef -self.learning_rate * gradient
    def fitBySgd(self,x,y):
        #stochastic gradient descent
        for i in range(self.max_iter):
            print('iterating...')
    def fitByMbsgd(self,x,y):
        #moment based gradient descent
        for i in range(self.max_iter):
            print('iterating...')
    def predict(self,input) :
        
        x=np.array(input)
        # add the intercept column
        x=np.hstack((np.ones((x.shape[0],1)),x))
        y_proba=self.hypo(x @ self.coef)
        y_predict = (y_proba > 0.5).astype(int)
        print(y_predict.shape)
        return y_predict
    def setHypo(self):
        if self.proba=='sigmoid':
            self.hypo=sigmoid
        elif self.proba=='tanh':
            self.hypo=tanh
