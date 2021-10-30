#jk
import numpy as np

class AllModelParams:
    
    def __init__(self):
        pass
    
    def get_model_params(self):
        all_params_dict = {

        'logistic_regression':{
        # solver - Algorithm to use in the optimization problem. Default is ‘lbfgs'
        # penalty -- Specify the norm of the penalty
        # C -- C parameter controls the penality strength, smaller values specify stronger regularization
                            'solver' : ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                            #'l1_ratio': [0.25, 0.75],
                            'C': np.logspace(-3,3,7),
                            'penalty': ['l1', 'l2', 'elasticnet']
         },
        'decision_tree':{
        # hyperparameter tuning
        # https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
        # https://www.projectpro.io/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python
        # criterion -- The function to measure the quality of a split
        # splitter -- The strategy used to choose the split at each node.
                        # for a tree with few features without any overfitting, I would go with the “best” splitter to be safe 
                        # so that you get the best possible model architecture
        # max_depth -- The maximum depth of the tree.
                        #In general, the deeper you allow your tree to grow, the more complex your model will become 
                        # because you will have more splits and it captures more information about the data and this is one 
                        # of the root causes of overfitting. if your model is overfitting, reducing the number for max_depth is one
                        # way to combat overfitting.Too low values can also lead to under-fitting
        # min_samples_split -- The minimum number of samples required to split an internal node.
                        # the ideal min_samples_split values tend to be between 1 to 40 for the CART algorithm which is the 
                        # algorithm implemented in scikit-learn. min_samples_split is used to control over-fitting
                        # Too high values can also lead to under-fitting
        # min_samples_leaf -- The minimum number of samples required to be at a leaf node
                             #leaf node is a node without any children.ideal min_samples_leaf values tend to be 
                             #between 1 to 20 for the CART algorithm
        # max_features -- The number of features to consider when looking for the best split
                          # If None, then max_features=n_features
                          # If “sqrt”, then max_features=sqrt(n_features).
                          # If “log2”, then max_features=log2(n_features).
                          # if you have a high computational cost or you have a lot of overfitting, you can try with “log2” 
                          # and depending on what that produces, you can either bring it slightly up using sqrt

        # class_weight -- Weights associated with classes in the form {class_label: weight}. If not given, 
                          # all classes are supposed to have weight one
                          # The "balanced" mode uses the values of y to automatically adjust weights inversely proportional 
                          # to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))
        #ccp_alpha -->   is a threshold, complexity of any branch more than this will be removed (we can use for overfitted model)
                         # pruning(post pruning) using ccp_alphaa..
        ### summary:
        #The Decision tree complexity has a crucial effect on its accuracy and it is explicitly controlled 
        #by the stopping criteria used and the pruning method employed. Usually, the tree complexity is measured 
        #by one of the following metrics: the total number of nodes, total number of leaves, 
        # tree depth and number of attributes used [8]. max_depth, min_samples_split, and min_samples_leaf are 
        #all stopping criteria whereas min_weight_fraction_leaf and min_impurity_decrease are pruning methods.
                     "criterion":['gini','entropy'],
                     "splitter":['best','random'],
                     "max_depth" : [2, 3, 5, 10, 20],
                     "min_samples_split":range(2,40,3),
                     "min_samples_leaf":range(1,10,2),
                     'ccp_alpha':np.random.rand(5),
                     'max_features': [None],
                     'class_weight': [None, 'balanced']
                     },
        'support_vector_classifier':{
        # https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
        # https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
        # C -- Regularization parameter. The strength of the regularization is inversely proportional to C. 
             #Must be strictly positive. The penalty is a squared l2 penalty.
            # C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary 
            #and classifying the training points correctly
            # Increasing C values may lead to overfitting the training data.
        # kernel -- Specifies the kernel type to be used in the algorithm
                  # kernel parameters selects the type of hyperplane used to separate the data.
                  # Using ‘linear’ will use a linear hyperplane (a line in the case of 2D data). \
                  # ‘rbf’ and ‘poly’ uses a non linear hyper-plane
        # gamma -- parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set
                "kernel":['linear', 'poly', 'rbf', 'sigmoid' ],
                'C':[0.01,0.1,1,10,100,200,500],
                #'gamma':[.001,.1,.4,.004,.003]
        }, 
        'k_nearest_neighbour':{
        # hyper parameter tuning
        # https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
        # https://www.kaggle.com/arunimsamudra/k-nn-with-hyperparameter-tuning
        # n_neighbors -- n_neighbors represents the number of neighbors to use for kneighbors queries
        # weights -- 'uniform' assigns no weight, while 'distance' weighs points by the inverse of their distances meaning 
           #nearer points will have more weight than the farther points.
        # metric -- The distance metric to be used will calculating the similarity.
        # p -- This is the power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance(l1),
              # and euliddean_distance(l2) for p=2. For arbitrary p, minkowski distance (l_p) is used
            'n_neighbors':[3,5,7,9,12,13,15,17,21],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            #'leaf_size' : [10 ,15, 20 , 25 , 30 , 35 , 45 , 50 ],
            #'p' : [1,2],
            'weights' : ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'bagging_decision_tree':{
        # hyper tuning parameter
        #https://stackoverflow.com/questions/47570307/tuning-parameters-of-the-classifier-used-by-baggingclassifier
                'n_estimators': [10,50,250,1000],
                'base_estimator__criterion':['gini','entropy'],
                'base_estimator__splitter':['best','random'],
                'base_estimator__max_depth' : [2, 3, 5, 10, 20],
                'base_estimator__min_samples_split':range(2,40,3),
                'base_estimator__min_samples_leaf':range(1,10,2),
                'base_estimator__ccp_alpha':np.random.rand(5),
                'base_estimator__max_features': [None],
                'base_estimator__class_weight': [None, 'balanced']
        }
        }
        return all_params_dict
    