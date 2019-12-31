import numpy as np
import pylab
import random

# parameters for perceptron and SVM
eps = 1e-10  # tolerance

def svm_standard(X,Y):
    """svm_margin(X,Y) returns the weights, bias and margin if the given pattern set X with labels Y is linearly separable, and 0s otherwise. Y vector should consist of 0 and 1 only"""
    w = np.zeros(X.shape[0])
    from sklearn import svm
    hyp = svm.SVC(kernel='linear',C=10000,cache_size=20000,tol=1e-5)
    hyp.fit(X.T,Y)
    b = hyp.intercept_[0]
    for j in range(hyp.support_.size):
        w += hyp.dual_coef_[0][j]*hyp.support_vectors_[j]
    dec = np.sign(np.dot(w.T,X)+b)
    dec[dec<0] = 0
    if abs(np.sum(np.abs(Y-dec))) > eps:
        return np.zeros(X.shape[1]),0,0
    else:
        return w,b,2./pylab.norm(w)

def svm_qp(x,y,is_bias=1,is_wconstrained=1):
    """svm_qp(x,y,is_bias=1,is_wconstrained=1) returns the weights, bias and margin if the given pattern set X with labels Y is linearly separable, and 0s otherwise. x is the input matrix with dimension N (number of neurons) by P (number of patterns). y is the desired output vector of dimension P. y vector should consist of -1 and 1 only"""
    import qpsolvers
    R = x.shape[1]
    G = -(x*y).T
    if is_bias:
        N = x.shape[0] + 1
        G = np.append(G.T,-y)
        G = G.reshape(N,R)
        G = G.T
        P = np.identity(N)
        P[-1,-1] = 1e-12    # regularization
    #for j in range(N):
    #P[j,j] += 1e-16
    #P += 1e-10
    else:
        N = x.shape[0]
        P = np.identity(N)
    if is_wconstrained:
        if is_bias:
            G = np.append(G,-np.identity(N)[:N-1,:])
            G = G.reshape(R+N-1,N)
            h = np.array([-1.]*R+[0]*(N-1))
        else:
            G = np.append(G,-np.identity(N))
            G = G.reshape(R+N,N)
            h = np.array([-1.]*R+[0]*N)
    else:
        h = np.array([-1.]*R)
    w = qpsolvers.solve_qp(P,np.zeros(N),G,h)
    if is_bias:
        return w[:-1],w[-1],2/pylab.norm(w[:-1])
    else:
        return w,2/pylab.norm(w)

"""
# Standard SVM examples
X = np.array([[1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1],
           [1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1]])
Y1 = np.array([1,1,0,0,0,0])
Y2 = np.array([1,0,1,0,0,0])
w,b,k = svm_standard(X,Y1)
print w,b,k
w,b,k = svm_standard(X,Y2)
print w,b,k
"""

# SVM in 2D for illustration
X = np.array([[1.,2,2.5,3,3,4],[5,3,2,-1,0,1]])
Y = np.array([0,0,0,1,1,1])
x = np.arange(np.min(X[0])-2,np.max(X[0])+2)
w,b,k = svm_standard(X,Y)
print w,b,k
pylab.figure()
pylab.plot(X[0][Y==0],X[1][Y==0],'bx')
pylab.plot(X[0][Y==1],X[1][Y==1],'ro')
if k > 0:
    pylab.plot(x,-w[0]/w[1]*x-b/pylab.norm(w)/np.sin(np.arctan(w[1]/w[0])),'k',label='standard sklearn')
else:
    print 'Standard SVM: Not linearly separable!'

Y[Y==0] = -1
w,b,k = svm_qp(X,Y,1,0)
print w,b,k
if k > 0:
    pylab.plot(x,-w[0]/w[1]*x-b/pylab.norm(w)/np.sin(np.arctan(w[1]/w[0])),'g--',label='standard QP')
else:
    print 'SVM: Not linearly separable!'

Y[Y==0] = -1
w,k = svm_qp(X,Y,0,0)
print w,k
if k > 0:
    pylab.plot([0],[0],'y.')
    pylab.plot(x,-w[0]/w[1]*x,'y--',label='no bias')
else:
    print 'SVM: Not linearly separable!'

Y[Y==0] = -1
w,b,k = svm_qp(X,Y,1,1)
print w,b,k
if k > 0:
    if w[1] == 0:
        pylab.plot([-b/w[0]]*2,[np.min(X[1]-2),np.max(X[1]+2)],'m--',label='weight constraint')
    else:
        pylab.plot(x,-w[0]/w[1]*x-b/pylab.norm(w)/np.sin(np.arctan(w[1]/w[0])),'m--',label='weight constraint')
else:
    print 'SVM: Not linearly separable!'
pylab.legend(loc=2)
pylab.show()
