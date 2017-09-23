from numpy import *
import pandas as pd
from sklearn import datasets

def compute_error_for_line_given_points(b,m,points):
    # initialize it at 0
    totalError = 0
    #for every points
    for i in range(0,len(points)):
        #get the x value
        x = points[i, 0]
        #get the y value
        y = points [i, 1]
        # get the difference , square it, add it to thr total
        totalError += (y - (m * x + b))**2
    # get the average
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iterations):
        # update b and m with the new more accurate b and m by performing this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    #starting points for the gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range (0, len(points)):
        x = points[i, 0]
        y = points [i, 1]
        #direction with respect to b and m
        #computing partial derivatives of the error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    # update b and m values using partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b , new_m]

def run():
    # step 1 - collect the data
    points  = genfromtxt('C:/Users/Home Premium/Desktop/boston_data_set.csv',delimiter = ',')
    print points

    learning_rate = 0.001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1500
    # step 3 - train the model
    print 'starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m,
                                                                              compute_error_for_line_given_points(initial_b, initial_m, points))
    [b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)
    print 'ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b,m, points))


run()


