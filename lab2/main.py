import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def lin_reg(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_train]
    theta_best = np.linalg.pinv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)

    t = list(x_test)
    t.sort()
    x_test = np.array(t)

    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_test]

    y_predict = x_new_b.dot(theta_best)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x_train, y_train, "b.")
    plt.show()
    return theta_best


def gradient_descent(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_train]

    eta = 0.1
    n_iteration = 1000
    m = int(size - 0.2 * size)
    theta = np.random.rand(2, 1)
    for iteration in range(n_iteration):
        gradients = 2 / m * x_b.T.dot(x_b.dot(theta) - y_train)
        theta = theta - eta * gradients

    t = list(x_test)
    t.sort()
    x_test = np.array(t)
    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_test]
    y_predict = x_new_b.dot(theta)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x_train, y_train, "b.")
    plt.show()
    return theta


def learning_shedule(t):
    t0, t1 = 5, 50
    return t0 / (t + t1)


def mini_batch_gradient_descent(x, y, size_batch):
    s = len(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_b = np.c_[np.ones((int(s - 0.2 * s), 1)), x_train]
    n_epochs = 50
    m = int(s - 0.2 * s)
    theta = np.random.rand(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(s - size_batch)
            xi = x_b[random_index:random_index + size_batch]
            yi = y_train[random_index:random_index + size_batch]
            gradients = 2 / size_batch * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_shedule(epoch * m + i)
            theta = theta - eta * gradients
    t = list(x_test)
    t.sort()
    x_test = np.array(t)
    x_new_b = np.c_[np.ones((int(0.2 * s), 1)), x_test]
    y_predict = x_new_b.dot(theta)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x_train, y_train, "b.")
    plt.show()
    return theta


def sgd(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_train]
    n_epochs = 50
    m = int(size - 0.2 * size)
    theta = np.random.rand(2, 1)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = x_b[random_index:random_index + 1]
            yi = y_train[random_index:random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_shedule(epoch * m + i)
            theta = theta - eta * gradients
    t = list(x_test)
    t.sort()
    x_test = np.array(t)

    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_test]
    y_predict = x_new_b.dot(theta)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x_train, y_train, "b.")
    plt.show()
    return theta


def pr(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly_features.fit_transform(x_train)
    x_b = np.c_[np.ones((int(size - 0.2 * size), 1)), x_poly]
    eta = 0.0017
    n_iteration = 10000
    m = size - 0.2 * size
    lamda = 0.08
    theta = np.random.rand(4, 1)
    for iteration in range(n_iteration):
        gradients = 1 / m * x_b.T.dot(x_b.dot(theta) - y_train) + lamda / m * theta
        theta = theta - eta * gradients

    t = list(x_test)
    t.sort()
    x_test = np.array(t)
    x_poly = poly_features.fit_transform(x_test)

    x_new_b = np.c_[np.ones((int(0.2 * size), 1)), x_poly]

    y_predict = x_new_b.dot(theta)
    plt.plot(x_test, y_predict, "r-")
    plt.plot(x, y, "b.")
    plt.show()
    polynomial_regression = Pipeline(
        [("poly_features", PolynomialFeatures(degree=3, include_bias=False)), ("lin_reg", LinearRegression()), ])
    plot_learning_curves(polynomial_regression, x, y)
    return theta


def plot_learning_curves(model, x, y):
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


def mse(x, y, thetta, c_lambda=0.08):
    sum_delta = 0
    sum_thetta = 0
    s = len(y)
    for i in range(s):
        sum_delta += (thetta.transpose().dot(x[i]) - y[i]) ** 2
    for j in range(1, len(thetta)):
        sum_thetta += thetta[j] ** 2
    return sum_delta / (2 * s) + c_lambda * sum_thetta / (2 * s)


def batch_gradient_descent_pr(x, y, eta=0.0017, lamda=0.08):
    s = len(y)
    n_iteration = 10000
    m = s - 0.2 * s
    theta = np.random.rand(4, 1)
    for iteration in range(n_iteration):
        gradients = 1 / m * x.T.dot(x.dot(theta) - y) + lamda / m * theta
        theta = theta - eta * gradients
    return theta


def plot_func_eta_j(x, y):
    j = []
    s = len(y)
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly_features.fit_transform(x)
    x_b = np.c_[np.ones((s, 1)), x_poly]
    alpha = np.arange(0.0005, 0.005, 0.0001)
    for i in alpha:
        thetta = batch_gradient_descent_pr(x_b, y, eta=i)
        j.append(mse(x_b, y, thetta))
    plt.plot(alpha, j, 'r-')
    plt.show()


size = 1000


def main():
    x2 = 5 * np.random.rand(size, 1) - 3
    y2 = x2 ** 3 + 2 * x2 ** 2 + 3 * x2 + 10 + np.random.rand(size, 1)
    # +-10, 3, 2, 1
    x = 3 * np.random.rand(size, 1)
    y = 4 + 3 * x + np.random.randn(size, 1)
    # y = 4 +3x + random
    print("____________________________________")
    print("Линейная регрессия:")
    print(lin_reg(x, y))
    print("____________________________________")
    print("Градиентный спуск:")
    print(gradient_descent(x, y))
    print("____________________________________")
    print("SGD:")
    print(sgd(x, y))
    print("____________________________________")
    print("mini:")
    print(mini_batch_gradient_descent(x, y, size // 100))
    print("____________________________________")
    print("Полиномиальная регрессия: ")
    print(pr(x2, y2))
    print("____________________________________")
    print(plot_func_eta_j(x2, y2))


if __name__ == "__main__":
    main()
