import numpy as np


class LinearRegression():

    def __init__(self, learning_rate=0.000000001):
        self.learning_rate = learning_rate

    def fit(self, X: np.array, y: np.array):
        self.m_examples = X.shape[0]
        self.n_features = X.shape[1]

        self.train_X = np.array([np.append(arr, np.array([1])) for arr in X]) # weights and biases at the same place
        self.train_y = y

        self.weights = self.get_init_weights()

        self.optimize()


    def predict(self, test_X: np.array):

        test_X = np.array([np.append(arr, np.array([1])) for arr in test_X])

        pred_results = self.hypoth(test_X)
        return pred_results

    def get_init_weights(self):

        theta = np.zeros(self.n_features)
        theta = np.append(theta, np.array([1]))[:, np.newaxis]

        return theta

    def compute_cost(self):
        return (1 / 2 * self.m_examples) * np.linalg.norm(self.hypoth(self.train_X) - self.train_y)

    def hypoth(self, features_table):
        return features_table.dot(self.weights)

    def compute_weights_gradient(self):
        try:
            gradient = (1 / self.m_examples) * self.train_X.T.dot(
                self.train_X.dot(self.weights) - self.train_y
            )
        except:
            breakpoint()

        return gradient


    def visualize(self):

        """
        For future visualization of the results
        :return:
        """
        pass

    def optimize(self):

        ### Gradient Descend
        # TODO: Change the stop criterion
        for i in range(5000):
            d_weights = self.compute_weights_gradient()
            self.weights -= self.learning_rate * d_weights

            if i % 100 == 0:
                print(i)

        print("Optimized!!!")
