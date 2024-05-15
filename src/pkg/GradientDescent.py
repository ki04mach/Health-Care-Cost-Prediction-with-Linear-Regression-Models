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
        self.y_train = y_train
        self.num_iter = num_iter
        self.alpha = alpha
        self.encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.b = np.mean(y_train)
        self.x_train = self._preprocess_data(x_train)
        self.w = np.random.rand(self.x_train.shape[1]) * 0.01
        self.n = len(self.x_train)
        if self.n == 0:
            raise ValueError("Training data cannot be empty.")
        if self.x_train.shape[1] == 0:
            raise ValueError("No features to train on after preprocessing.")

    def _preprocess_data(self, x: pd.DataFrame, fit_transform=True):
        """
        Preprocesses the data by applying one-hot encoding to categorical variables and z-score normalization
        to numerical variables. It adapts the processing based on whether it's fitting and transforming training
        data or just transforming test/prediction data.

        Args:
        x (pd.DataFrame): The DataFrame to preprocess.
        fit_transform (bool): If True, fits the encoder and normalization parameters; if False, uses pre-fitted parameters.

        Returns:
            pd.DataFrame: The preprocessed and normalized training data, ready for gradient descent or prediction.

        Notes:
            This method updates the encoder and normalizer during the fit_transform phase and uses the fitted parameters
            to transform data when fit_transform is False.
        """
        categorical_cols = x.select_dtypes(include=['object']).columns
        numerical_cols = x.select_dtypes(include=[np.number]).columns
        if len(categorical_cols) > 0:
            if fit_transform:
                self.encoder.fit(x[categorical_cols])
                
            encoded_cats = self.encoder.transform(x[categorical_cols])
            encoded_cats_df = pd.DataFrame(encoded_cats, 
                                            columns=self.encoder.get_feature_names_out(categorical_cols),
                                            index=x.index)
        normalized_data = self._zscore_normalization(x[numerical_cols], apply_fit=fit_transform)
        x_1 = pd.DataFrame(normalized_data, columns=numerical_cols)
        x_processed = pd.concat([x_1[numerical_cols], encoded_cats_df], axis=1)
        return x_processed
    
    def _zscore_normalization(self, df, apply_fit=True):
        """
        Normalizes numerical features in the DataFrame to have zero mean and unit variance. This method can fit and
        transform the data (calculate new mean and std) or just transform it using existing parameters.

        Args:
            df (pd.DataFrame): Numerical data to normalize.
            apply_fit (bool): Whether to fit new parameters (mean, std) or use existing ones (default: True).

        Returns:
            x_norm (pd.DataFrame): normalized training data
        Notes:
            Modifies `self.mean` and `self.std` when `apply_fit` is True.
        """
        if apply_fit:
            self.mean = df.mean(axis=0)
            self.std = df.std(axis=0)
        x_norm = (df - self.mean) / self.std
        return x_norm
    
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
            if np.allclose(self.dj_dw, 0, atol=1e-6) and np.allclose(self.dj_db, 0, atol=1e-6):
                print(f"Convergence at iteration {i:5}")
                break
            self.w -= self.alpha * self.dj_dw
            self.b -= self.alpha * self.dj_db
            if i % (self.num_iter // 10) == 0 or i == self.num_iter - 1:
                print(f"Iteration {i:5}: Cost {self.cost}")
        print(f"Final value of w is {self.w} and b is {self.b}")
    
    def predict(self, x_test):
        """
        Predicts the target variable for given input features using the trained model weights and bias.

        Args:
            x_test (pd.DataFrame): Test data features for which to make predictions.

        Returns:
            np.array: Predicted values based on the model.

        Notes:
            The input `x_test` must be preprocessed using the same steps as the training data to ensure consistency.
        """
        x_test_pre = self._preprocess_data(x_test, fit_transform=False)
        y_hat = x_test_pre.dot(self.w) + self.b
        return y_hat
