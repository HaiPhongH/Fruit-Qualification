from __future__ import print_function
import numpy as np

def get_coeff(class1, class2, labels1, labels2):
    
    N = len(class1)
    X = np.concatenate((class1, class2), axis = 0)
    y = np.concatenate((labels1, labels2), axis = 0) # label
    # solving the dual problem (variable: lambda)
    from cvxopt import matrix, solvers
    V = np.concatenate((class1, -class2), axis = 0) # V in the book
    Q = matrix(V.dot(V.T))
    p = matrix(-np.ones((2*N, 1))) # objective function 1/2 lambda^T*Q*lambda - 1^T*lambda
    # build A, b, G, h
    G = matrix(-np.eye(2*N))
    h = matrix(np.zeros((2*N, 1)))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros((1, 1)))
    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)
    l = np.array(sol['x']) # solution lambda
    # calculate w and b

    w = V.T.dot(l)
    S = np.where(l > 1e-8)[0] # support set, 1e-8 to avoid small value of l.
    b = np.mean(y[S].reshape(-1, 1) - X[S,:].dot(w))
    print('Number of suport vectors = ', S.size)
    print('w = ', w.T)
    print('b = ', b)