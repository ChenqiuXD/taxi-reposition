from sklearn.neighbors import KNeighborsRegressor
import numpy as np
 
def train_test_split(sensor_output, input_length, output_length=1, test_size=0.2):
    """
        Split the dataset into training and testing sets.
        Return: 
            X_train: training input, [n_samples, input_length, n_features]
            X_test: testing input, [n_samples, input_length, n_features]
            y_train: training output, [n_samples, output_length, n_features]
            y_test: testing output, [n_samples, output_length, n_features]
    """
    n_features = sensor_output.shape[0]
    n_samples = sensor_output.shape[1] - input_length - output_length + 1

    X = np.zeros([n_samples, input_length, n_features])
    Y = np.zeros([n_samples, output_length, n_features])
    for i in range(n_samples):
        X[i, :, :] = sensor_output[:, i:i+input_length]
        Y[i, :, :] = sensor_output[:, i+input_length:i+input_length+output_length].T

    X_train = X[:int(n_samples*(1-test_size)), :, :]
    X_test = X[int(n_samples*(1-test_size)):, :, :]
    y_train = Y[:int(n_samples*(1-test_size)), :, :]
    y_test = Y[int(n_samples*(1-test_size)):, :, :]

    assert X_train.shape[0] == y_train.shape[0]

    return X_train, X_test, y_train, y_test

# Load the dataset
x = np.tile(np.linspace(0, 5, 500), [3,1])
sensor_output = np.sin(x) + np.array([[1], [2], [3] ])
sensor_output += np.random.normal(0, 0.1, sensor_output.shape)
X_train, X_test, y_train, y_test = train_test_split(sensor_output, input_length=3, output_length=1, test_size=0.2)
 
# Apply KNN regression
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train.reshape([X_train.shape[0], -1]), y_train.reshape([y_train.shape[0], -1]))
predictions = knn_regressor.predict(X_test.reshape([X_test.shape[0], -1]))
 
# Evaluate the model
print('Score:', knn_regressor.score(X_test.reshape([X_test.shape[0], -1]), y_test.reshape([y_test.shape[0], -1 ])))

print('Predictions:\n', predictions[:10])
print('Ground truth:\n', y_test.reshape([y_test.shape[0], -1])[:10])
