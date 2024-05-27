import numpy as np
import matplotlib.pyplot as plt

def simple_linear_regression(x, y):
    # Calculate the mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Total number of values
    n = len(x)
    
    # Using the formula to calculate b1 and b0
    numer = 0
    denom = 0
    for i in range(n):
        numer += (x[i] - mean_x) * (y[i] - mean_y)
        denom += (x[i] - mean_x) ** 2
    b1 = numer / denom
    b0 = mean_y - (b1 * mean_x)
    
    # These are the coefficients
    return (b0, b1)

def plot_regression_line(x, y, b):
    # Plotting the actual points as scatter plot
    plt.scatter(x, y, color="m", marker="o", s=30)
    
    # Predicted response vector
    y_pred = b[0] + b[1] * x
    
    # Plotting the regression line
    plt.plot(x, y_pred, color="g")
    
    # Putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Function to show plot
    plt.show()

# Driver code
if __name__ == "__main__":
    # Data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    
    # Estimating coefficients
    b = simple_linear_regression(x, y)
    print("Estimated coefficients:\nb0 = {:.2f}\nb1 = {:.2f}".format(b[0], b[1]))
    
    # Plotting regression line
    plot_regression_line(x, y, b)
