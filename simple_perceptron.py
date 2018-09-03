""" Coding a perceptron """

def perceptron():
    
    # Importing libraries
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initializing random seed for reperforming calculations
    np.random.seed(74383)
    
    # Activation function
    def activ(ans):
        if ans>0:
            return 1
        else:
            return -1
    
    # Setting up a line as a classification boundary
    x = [x for x in range(-3,15)]
    y = np.linspace(20,-1,18)
    
    
    # Function to return y-coordinate for corresponding x-coordinate of line
    def line_ycor(m):
        n = np.round(((y[5] - y[0])/(x[5] - x[0])),2)*(m - x[0]) + y[0]
        return n
    
    
    # Returns whether point (a,b) is above or below the line
    def abv_or_bel(a, b):
        if b > line_ycor(a):
            return 1
        else:
            return -1
        
    
    # Initializing inputs and bias
    x1 = [12.5,10]
    b = 1
    x1.append(b)
    x1 = np.array(x1)
    
    
    # Checking whether point lies above or below line
    if abv_or_bel(x1[0], x1[1])==1:
        x20 = 1 
        print('Point lies above line')
    else:
        x20 = -1
        print('Point lies below line')
    
    
    # Plotting the line and input point(s) and displaying
    plt.plot(x,y)
    plt.scatter(x1[0], x1[1], c='r')
    plt.show()
    
    
    # Initializing weights for corresponding inputs and bias
    w = np.array([np.random.choice([-1,1])*np.random.rand() for x in range(3)])
    
    # Intializing error to a non-zero value and count to 1
    err = 1
    count = 1
    
    while(err):
        print('Iteration %d'%count)
        # Calculating output and applying activation function
        x2 = np.sum(x1*w)
        x2 = activ(x2)
        # Printing the output obtained and the output desired
        print('The obtained value is: %d'%x2)
        print('The desired value is: %d'%x20)
        
        """ Desired value (DV) = x20, Obtained value (OV) = x2
        
        So, Error (err) = AV - OV = x20 - x2
        
        delta_weight (del_w) = err*x1*c (where c => Learning Rate)
        
        New Weight (nw) = w + del_w = w + err*x1*c
        
        x1 = [ x1[0] , x1[1] , b ]
        
        w = [w1, w2, w3]
          = [ w[0] , w[1] , w[3] ]
        
        del_w = [del_w1, del_w2, del_w3]
              = [ err * x1[0] * c  ,  err * x[0] * c  ,  err * b * c ]
              
        nw = [nw1, nw2, nw3] = [ w1 + del_w1  ,  w2 + del_w2  ,  w3 + del_w3 ]
        
        Weight values are replaced by New Weight values i.e., w = nw
        
        The process of evaluating the output is then continued 
        until Error = 0 """
        
        err = x20 - x2
        print('The error is: %d'%err)
        print()
        
        c = 0.04
        
        del_w = err*x1*c
    
        nw = w + del_w
        
        w = nw
        
        count += 1