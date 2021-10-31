import numpy as np
import matplotlib.pyplot as plt

#generates x and y numpy arrays for 
# y = a*x + b + a * noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_linear(a, b, noise, filename, size = 100):
    print('Generating random data y = a*x + b')
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b +  noise*a*(np.random.rand(size, 1) -0.5)
    data = np.hstack((x,y))
    np.savetxt(filename,data,delimiter=',')
    return(x,y)




# thats an example of linear regression using polyfit
def linear_regression_numpy(filename):
    # now let's read it back
    with open(filename, 'r') as f:
        data = np.loadtxt(f,delimiter=',')
    #split to initial arrays
    x,y = np.hsplit(data,2)
    #printing shapes is useful for debugging
    print(np.shape(x))
    print(np.shape(y))
    #our model
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    # our hypothesis for give x
    h =  model[0]*x + model[1]

    #and check if it's ok
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label = 'experiment')
    plt.plot(x, h, "r", label = 'model')    
    plt.legend()
    plt.show()
    


#generates x and y numpy arrays for 
# y = a_n*X^n + ... + a2*x^2 + a1*x + a0 + noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_poly(a, n, noise, filename, size = 100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size,1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n+1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0,n+1):
        y = y + a[i] * np.power(x,i) + noise*(np.random.rand(size, 1) -0.5)
    print(np.shape(x))
    data = np.hstack((x,y))
    np.savetxt('generated_poly.csv',data,delimiter=',')
    # now let's read it back
    with open('generated_poly.csv', 'r') as f:
        data = np.loadtxt(f,delimiter=',')
    #split to initial arrays
    x,y = np.hsplit(data,2)
    #and check if it's ok
    plt.title(f"Polynomial regression task (n={n})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label = 'experiment')
    plt.legend()
    plt.show()
    return(x,y)

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) partial derivatives of J over theta (shape is 1 x N - the same as theta)

def gradient_descent_step(J, theta, alpha):
    print("your code goes here")

    return(theta)

def get_J(x, y, theta):
    print("your code goes here")
    return 0   

def minimize(J, theta, paha):
    print("your code goes here")
    return


if __name__ == "__main__":
    generate_linear(1,-3,0.5,'linear',100)
    linear_regression_numpy("linear")
    
    print('Generating polynomial data y = a*x^2 + bx + c')
    (x,y) = generate_poly([1,2,3],2,0.5)
    
    