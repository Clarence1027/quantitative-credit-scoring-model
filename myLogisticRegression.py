import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


class LogisticRegression:
    '''Logistic Regression for binary classification'''

    def __init__(self, fitIntercept=True, C=1, tol=1e-4, maxIter=5000, randomState=0):
        self.fitIntercept = fitIntercept
        self.C=C
        self.tol = tol
        self.maxIter = maxIter
        self.randomState = randomState

    @staticmethod
    def linTransform(X, beta):
        '''
        Linear transformation before logistic activation.

            Param:
            -------
            @X: input features, shape=(n,m), where n is # of samples, m is # of features
            @beta: linear coefficients, shape=(m,1) or (m,)

            Return:
            --------
            The logit that will be activated using sigmoid
        '''
        return np.dot(X, beta)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def crossEntropyLoss(proba, trueLabel):
        '''
        Cross entropy loss for binary classification, which is also the log likelihood of given samples.

            Param:
            -------
            @proba: probability (of being positive) predicted from model
            @trueLabel: true label of samples

            Return:
            --------
            Cross entropy loss
        '''
        if proba.shape != trueLabel.shape:
            raise Exception('shape not match for prediction and truth')

        return -np.sum(trueLabel * np.log(proba) + (1 - trueLabel) * np.log(1 - proba)) / trueLabel.shape[0]

    def train(self,Xtrain,Ytrain):
        '''
        Train logistic regression.

        :param Xtrain: shape=(n,m) where n = # of samples, m = # of features
        :param Ytrain: shape=(n,)

        :return: logistic model that's trained
        '''

        # random initialization
        np.random.seed(self.randomState)
        if self.fitIntercept:
            beta=np.random.randn(Xtrain.shape[1]+1)*0.0001
            #beta=np.zeros((Xtrain.shape[1]+1,))
            X=np.hstack((np.ones((Xtrain.shape[0],1)),Xtrain))
        else:
            beta = np.random.randn(Xtrain.shape[1]) * 0.0001
            #beta=np.zeros((Xtrain.shape[1],))
            X=Xtrain.copy()

        epoch=0
        self.loss=[]
        while epoch < self.maxIter:
            # forward propagate
            z=LogisticRegression.linTransform(X,beta)
            proba=LogisticRegression.sigmoid(z)
            loss=LogisticRegression.crossEntropyLoss(proba,Ytrain) + self.C*np.dot(beta.T,beta)/2
            self.loss.append(loss)

            # gradient and hessian
            dB=np.dot(X.T,proba-Ytrain)/Ytrain.shape[0] + self.C*beta
            diag=np.diag(proba*(1-proba))
            hessian=X.T.dot(diag).dot(X) + self.C*np.eye(beta.shape[0])

            # update
            betaNew = beta - np.linalg.inv(hessian).dot(dB)

            if np.abs(betaNew-beta).max() < self.tol:
                break

            else:
                beta=betaNew
                epoch += 1

        self.beta=betaNew
        z=LogisticRegression.linTransform(X,self.beta)
        proba=LogisticRegression.sigmoid(z)
        loss=LogisticRegression.crossEntropyLoss(proba,Ytrain) + self.C*np.dot(self.beta.T,self.beta)/2
        self.loss.append(loss)
        self.hessian=X.T.dot(np.diag(proba*(1-proba))).dot(X) + self.C*np.eye(self.beta.shape[0])

        if epoch == self.maxIter:
            print('optimization not converged')


    def predict(self,Xtest):
        if self.fitIntercept:
            X=np.hstack((np.ones((Xtest.shape[0],1)),Xtest))
        else:
            X=Xtest.copy()
        z=LogisticRegression.linTransform(X,self.beta)
        proba=LogisticRegression.sigmoid(z)

        return proba

    def print_loss_curve(self):
        plt.plot(self.loss)
        plt.show()

    def clear(self):
        try:
            del self.beta
            del self.loss
            del self.hessian
        except:
            pass



class stepwiseFwdLR(LogisticRegression):
    def __init__(self, fitIntercept=False, tol=1e-4, C=1, maxIter=5000, randomState=0):
        super().__init__(fitIntercept,C,tol,maxIter,randomState)

    def train(self,Xtrain,Ytrain):
        X = np.hstack((np.ones((Xtrain.shape[0], 1)), Xtrain))
        nSample = Xtrain.shape[0]
        included = [0]
        entire = list(np.arange(X.shape[1]))

        while True:
            rest = np.setdiff1d(entire, included)
            if rest is None:
                break

            pvalue = []
            AIC = []
            for idx in rest:
                super().clear()
                super().train(X[:, included + [idx]], Ytrain)

                tv = np.sqrt(nSample) * self.beta[-1] / np.sqrt(np.linalg.inv(self.hessian)[-1, -1])
                pvalue.append(2 * (1 - t.cdf(abs(tv), nSample - 1)))
                AIC.append(self.loss[-1])

            fid = rest[np.argmin(AIC)]
            p = pvalue[np.argmin(AIC)]

            if p < 0.05:
                included += [fid]
            else:
                break

        super().clear()
        super().train(X[:, included], Ytrain)

        return included

    def predict(self,Xtest):
        X=np.hstack((np.ones((Xtest.shape[0],1)),Xtest))
        z=LogisticRegression.linTransform(X,self.beta)
        proba=LogisticRegression.sigmoid(z)

        return proba
