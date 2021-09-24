import numpy as np
np.set_printoptions(suppress=True)
from sklearn import datasets
from sklearn.model_selection import train_test_split
def sig(z):
    return 1/(1 + np.exp(-z))
def derivativeSig(z):
    return sig(z)*(1 - sig(z))
def gd(X, Y, weights, bias, lr, num_iterations):
    for k in range(num_iterations):
        output0=X
        output=sig(np.dot(output0, weights)+bias)
        first_term=output-Y
        second_term=derivativeSig(np.dot(output0, weights)+bias)
        first_two = np.dot(first_term,second_term)
        changes=np.dot(output0.T, first_two)
        weights=weights-lr*changes
        bias_change=np.sum(first_two)
        bias=bias-lr*bias_change
    return weights, bias
def run():
    data=datasets.load_breast_cancer()
    X_data = data.data
    Y_data = data.target
#     print(X.shape, Y.shape)
    X,X_test, Y, Y_test=train_test_split(X_data, Y_data)
    weights=2*np.random.random((X.shape[1],1))-1
    bias=2*np.random.random(1)-1
    lr=2.5
    num_iterations=5000
    weights, bias=gd(X, Y, weights, bias, lr, num_iterations)
    Y_pred=sig(np.dot(X_test, weights)+bias)
    Y_pred=np.reshape(Y_pred,(143,))
    temp=abs(Y_pred-Y_test)
    print(temp.shape[0]-np.sum(temp), temp.shape[0])
run()