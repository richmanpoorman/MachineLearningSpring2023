# Matt & Bill
#
# QUESTIONS FOR JOE ROBINSON
#       Does our fit function have the right idea?
#       What is our theshold for the fit function?

import numpy as np
from scipy.optimize import minimize


# NOTE: follow the docstrings. In-line comments can be followed, or replaced.
#       Hence, those are the steps, but if it does not match your approach feel
#       free to remove.

def linear_kernel(X1, X2):
    """    Matrix multiplication.

    Given two matrices, A (m X n) and B (n X p), multiply: AB = C (m X p).

    Recall from hw 1. Is there a more optimal way to implement using numpy?
    :param X1:  Matrix A
    type       np.array()
    :param X2:  Matrix B
    type       np.array()

    :return:    C Matrix.
    type       np.array()
    """

    return np.dot(X1, X2)


def nonlinear_kernel(X1, X2, sigma=0.5):
    """
     Compute the value of a nonlinear kernel function for a pair of input vectors.

     Args:
         X1 (numpy.ndarray): A vector of shape (n_features,) representing the first input vector.
         X2 (numpy.ndarray): A vector of shape (n_features,) representing the second input vector.
         sigma (float): The bandwidth parameter of the Gaussian kernel.

     Returns:
         The value of the nonlinear kernel function for the pair of input vectors.

     """
    # (Bonus) TODO: implement 

    # Compute the Euclidean distance between the input vectors
    # Compute the value of the Gaussian kernel function
    # Return the kernel value
    return None


def objective_function(X, y, a, kernel):
    """
    Compute the value of the objective function for a given set of inputs.

    Args:
        X (numpy.ndarray): An array of shape (n_samples, n_features) representing the input data.
        y (numpy.ndarray): An array of shape (n_samples,) representing the labels for the input data.
        a (numpy.ndarray): An array of shape (n_samples,) representing the values of the Lagrange multipliers.
        kernel (callable): A function that takes two inputs X and Y and returns the kernel matrix of shape (n_samples, n_samples).

    Returns:
        The value of the objective function for the given inputs.
    """
    # TODO: implement
    
    nSamples = y.shape[0]

    # Reshape a and y to be column vectors
    y = y.reshape(-1, 1)
    a = a.reshape(-1, 1)
    # Compute the value of the objective function
    # The first term is the sum of all Lagrange multipliers
    # The second term involves the kernel matrix, the labels and the Lagrange multipliers
    alphaSum = sum(a)

    # secondSum = 0
    # for i in range(0, nSamples):
    #     for j in range(i + 1, nSamples):
    #         secondSum += a[i] * a[j] * y[i] * y[j] * (kernel(X[i], X[j]))

    # print(a.T)

    # STUFF
    secondSum = np.sum( (kernel(X, X.T)) * (a @ a.T) * (y @ y.T) )
    # secondSum = np.sum( (kernel(X, X.T)) * np.matmul(a, a.T) * np.matmul(y, y.T) )

    return alphaSum - 0.5 * secondSum


class SVM(object):
    """
         Linear Support Vector Machine (SVM) classifier.

         Parameters
         ----------
         C : float, optional (default=1.0)
             Penalty parameter C of the error term.
         max_iter : int, optional (default=1000)
             Maximum number of iterations for the solver.

         Attributes
         ----------
         w : ndarray of shape (n_features,)
             Coefficient vector.
         b : float
             Intercept term.

         Methods
         -------
         fit(X, y)
             Fit the SVM model according to the given training data.

         predict(X)
             Perform classification on samples in X.

         outputs(X)
             Return the SVM outputs for samples in X.

         score(X, y)
             Return the mean accuracy on the given test data and labels.
         """

    def __init__(self, kernel=nonlinear_kernel, C=1.0, max_iter=1e3):
        """
        Initialize SVM

        Parameters
        ----------
        kernel : callable
          Specifies the kernel type to be used in the algorithm. If none is given,
          ‘rbf’ will be used. If a callable is given it is used to pre-compute 
          the kernel matrix from data matrices; that matrix should be an array 
          of shape (n_samples, n_samples).
        C : float, default=1.0
          Regularization parameter. The strength of the regularization is inversely
          proportional to C. Must be strictly positive. The penalty is a squared l2
          penalty.
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.a = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vectors, where n_samples is the number of samples and n_features 
          is the number of features. For kernel=”precomputed”, the expected shape 
          of X is (n_samples, n_samples).

        y : array-like of shape (n_samples,)
          Target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
          Fitted estimator.
        """
        # save alpha parameters, weights, and bias weight

        
        # TODO: Define the constraints for the optimization problem
        
        # constraints = ({'type': 'ineq', 'fun': ...},
        #                {'type': 'eq', 'fun': ...})
        
          # Checks for the minimum value (if there is less than zero, then fails the contraint)
        # greaterThanZero = lambda alpha: min(alpha)
          # Checks for the sum of alpha and y to equal 0
        # sumWithYEqualsZero = lambda alpha: sum(alpha * y)
        
        constraints = ({'type': 'ineq', 'fun': lambda a: a}, 
                       {'type': 'eq', 'fun': lambda a: np.dot(a, y)})
        # TODO: Use minimize from scipy.optimize to find the optimal Lagrange multipliers
        
        # res = minimize(...)
        # self.a = ...
        alphaObjectiveFunction = lambda alpha: -objective_function(X, y, alpha, self.kernel)
        initialGuess = np.zeros(y.shape)

        # options = {"maxiter": self.max_iter}
        res = minimize(alphaObjectiveFunction, 
                        x0 = initialGuess, 
                        # method = 'trust-constr', 
                        # hess = lambda x: np.zeros((len(initialGuess), len(initialGuess))),
                        constraints = constraints
                        # , options = options
                        )
        self.a = res.x

        print(res.message)
        # TODO: Substitute into dual problem to find weights

        # for i in range(len(self.a)):
        #     if (self.a[i] <= 1e-8):
        #         self.a[i] = 0
        
        # # self.w = ...
        # self.w = np.dot( self.a * y, X )
        
        self.w = np.zeros( X.shape[1] )
        for row in range( X.shape[0] ):
            if self.a[row] >= 1e-8:
                self.w += self.a[row] * y[row] * X[row]


        # print(self.w)
        # TODO: Substitute into a support vector to find bias

        # print( X[y == 1])
        # print( self.w.reshape(-1, 1))
        
        # self.b = ...
        self.b = -0.5 * (max((np.dot(X[y == -1], self.w.T))) + min(np.dot(X[y == 1], self.w.T)))
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        # TODO: implement

        weights = self.w

        result = np.matmul(X, weights)

        # assumes that result is a 1d array.
        normalized = np.array([1 if x >= -self.b else -1 for x in result])

        return normalized

    def outputs(X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        # TODO: implement

        return None

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels. 

        In multi-label classification, this is the subset accuracy which is a harsh 
        metric since you require for each sample that each label set be correctly 
        predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          True labels for X.

        Return
        ------
        score : float
          Mean accuracy of self.predict(X)
        """
        # TODO: implement

        return None
