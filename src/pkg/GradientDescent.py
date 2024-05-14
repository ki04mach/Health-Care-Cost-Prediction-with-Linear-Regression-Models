import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class RegressionGD:
    """
    A class to model Linear Regressions with Gradient Descent
    """
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, num_iter=1000, alpha=0.01):
        """
        Initializes the RegressionGD class with training data, target values, and gradient descent parameters.

        Args:
        x_train (pd.DataFrame): The input features dataframe, possibly containing both numerical and categorical data.
        y_train (pd.Series): The target variable series corresponding to the input features.
        num_iter (int): The number of iterations to run the gradient descent algorithm (default 1000).
        alpha (float): The learning rate, determining the step size at each iteration (default 0.01).

        Side Effects:
        Initializes weights and bias, preprocesses the training data, and sets up internal parameters for tracking progress.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.num_iter = num_iter
        self.alpha = alpha
        self.b = np.mean(y_train)
        self.x_train = self._preprocess_data()
        self.w = np.random.rand(self.x_train.shape[1]) * 0.01
        self.n = len(self.x_train)
        if self.n == 0:
            raise ValueError("Training data cannot be empty.")
        if self.x_train.shape[1] == 0:
            raise ValueError("No features to train on after preprocessing.")

    def _preprocess_data(self):
        """
        Preprocesses the training data by detecting categorical variables, applying one-hot encoding,
        and performing z-score normalization on all features.

        Returns:
            pd.DataFrame: The preprocessed and normalized training data, ready for gradient descent.

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
        self.x_train[numerical_cols] = self._zscore_normalization(self.x_train[numerical_cols])
        x_processed = pd.concat([self.x_train[numerical_cols], encoded_cats_df], axis=1)
        return x_processed
    
    def _cost_fun(self):
        """
        Computes the cost function of estimated w and b

        Returns:
            cost(float): The cost of using w and b as parameters of regression model    
        """
        f_wb =self.x_train.dot(self.w) + self.b
        self.diff = f_wb - self.y_train
        self.cost = (1 / (2 * len(self.x_train))) * np.sum(self.diff ** 2)

    def _gradient(self):
        """
        Computes the gradient of the cost function with respect to the model parameters (weights and bias).
        This method updates the internal state with the latest gradients.

        Side Effects:
            Updates the internal variables 'self.dj_dw' and 'self.dj_db' with the derivatives of the cost
            with respect to weights and bias, respectively.
        """
        self.dj_dw = self.x_train.T.dot(self.diff) / self.n
        self.dj_db = np.sum(self.diff) / self.n
    def _zscore_normalization(self, df):
        """
        Calculates z score nomalized values of each feature and instance

        Args:
            df (pd.DataFrame): traininng data for normalization

        Returns:
            x_norm (pd.DataFrame): normalized training data 
        """
        mu = df.mean(axis=0)
        sigma = df.std(axis=0)
        x_norm = (df - mu) / sigma
        return x_norm
    def gradient_descent(self):
        """
        Executes the gradient descent algorithm to fit the model parameters (weights and bias) to the training data.

        Returns:
            Tuple[pd.Series, float]: A tuple containing the final weights (pd.Series) and bias (float) after training.

        Side Effects:
            Iteratively updates weights and bias based on the gradient descent calculations, and prints the cost
            periodically for monitoring progress. May also print convergence information if early stopping is triggered.
        """
        for i in range(self.num_iter):
            self._cost_fun()
            self._gradient()
            if np.allclose(self.dj_dw, atol=1e-6) and np.allclose(self.dj_db, atol=1e-6):
                print(f"Convergence at iteration {i:5}")
                break
            self.w -= self.alpha * self.dj_dw
            self.b -= self.alpha * self.dj_db
            if i % (self.num_iter // 10) == 0 or i == self.num_iter - 1:
                print(f"Iteration {i:5}: Cost {self.cost}")
        return self.w, self.b