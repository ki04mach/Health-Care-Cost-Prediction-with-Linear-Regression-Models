import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class LinearGD:
    """
    A class to model Linear Regressions with Gradient Descent
    """
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, w: pd.Series, b: float, num_iter, alpha):
        self.x_train = x_train
        self.y_train = y_train
        self.w = w
        self.b = b
        self.n = len(self.x_train)
        self.num_iter = num_iter
        self.alpha = alpha

    def _preprocess_data(self):
        """
        Automatically detects categorical variables and applies one-hot encoding.
        Keeps numerical variables unchanged.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: A new DataFrame with one-hot encoded categorical variables
                            and original numerical variables.
        """
        categorical_cols = self.x_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return self.x_train
        numerical_cols = self.x_train.select_dtypes(include=[np.number]).columns
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_cats = encoder.fit_transform(self.x_train[categorical_cols])
        encoded_cats_df = pd.DataFrame(encoded_cats, 
                                        columns=encoder.get_feature_names(categorical_cols),
                                        index=self.x_train.index)
        x_processed = pd.concat([self.x_train[numerical_cols], encoded_cats_df], axis=1)

        return x_processed
    
    def _cost_fun(self):
        """
        Computes the cost function

        Args:
            x(pd.DataFrame): training data
            y(pd.DataFrame): target values
            w, b(Scalar): model parameters

        Returns:
            cost(float): The cost of using w and b as parameters of linear regression model    
        """
        self.f_wb =self.x_train.dot(self.w) + self.b
        self.diff = self.f_wb - self.y_train
        self.cost = (1 / (2 * len(self.x_train))) * np.sum(self.diff ** 2)
    def _gradient(self):
        """
        Computes the gradient(derivative)

        Args:
            x(pd.Dataframe): training data
            y(pd.Series): target values
            w (pd.Series): model parameter
            b(Scalar): model parameters

        Returns:
            dj_dw(pd.Series): derivative of cost with respect to w of liniear regression 
            dj_db(Scalar): derivatives of cost with respect to b of linear regression model
        """
        self.dj_dw = self.x_train.T.dot(self.diff) / self.n
        self.dj_db = np.sum(self.diff) / self.n
    def _zscore_normalization(self):
        """
        Calculates z score nomalized values of each feature and instance

        Args:
            x (pd.DataFrame): traininng data for normalization

        Returns:
            x_norm (pd.DataFrame): normalized training data 
        """
        mu = self.x_train.mean(axis=0)
        sigma = self.x_train.std(axis=0)
        x_norm = (self.x_train - mu) / sigma
        return x_norm
    def gradient_descent(self):
        """
        Performs gradient descent to fit w and b. Updates w and b 
        by taking num_iters gardient steps with learning rate alpha

        Args:
            x(pd.Series): training data
            y(pd.Series): target values
            w, b(Scalar): initial model parameters
            alpha(float): learning rate
            num_iter(int): number of iteration
            cost_fun(function): to calculate cost function
            gradient(function): to calculate derivatives

        Returns:
            w, b(Scalar): updated values of parameters after runnign gradient descent
        """
        self.x_train = self._preprocess_data()
        for i in range(self.num_iter):
            self._gradient()
            if np.allclose(self.dj_dw, 0) and np.allclose(self.dj_db, 0):
                print(f"Convergence at iteration {i:5}")
                break
            self.w -= self.alpha * self.dj_dw
            self.b -= self.alpha * self.dj_db
            if i % (self.num_iter // 10) == 0 or i == self.num_iter - 1:
                self._cost_fun()
                print(f"Iteration {i:5}: Cost {self.cost}")
        return self.w, self.b