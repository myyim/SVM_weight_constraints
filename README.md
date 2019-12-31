# SVM_weight_constraint

This Python code returns the weights, bias (if any) and margin using more general support-vector machine (SVM) with optional bias and weight constraints.

# Introduction
Support vector machine (SVM) is a binary classifer with optimal hyperplane that separates the two classes of linear separable patterns. The standard implementation involves a bias and weight constraints, and can be performed using sklearn. However, in reality, bias may not exist, and weights may be constrained by the neuron types (excitatory and inhibitory neurons). This more general SVM resolves those questions. See svm.pdf for details.  

# More general SVM with optional bias and weight constraints
svm_standard performs the standard SVM using sklearn package. svm_qp implements the more general SVM with optional bias and weight constraints. Examples are given for illustration.

# License
This project is licensed under the MIT License.
