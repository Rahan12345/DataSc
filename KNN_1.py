import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

np.random.seed(145)


""" Point to be classified """
p1 = float(input('Enter x-coordinate value: '))
p2 = float(input('Enter y-coordinate value: '))
p = np.array([p1,p2])
print()


""" Generating synthetic data """
def gen_synth_data(n):
    """ Concatenations of two Normal distribution matrices with n rows and 2 columns """ 
    points = np.concatenate((ss.norm(1,1).rvs((n,2)),ss.norm(2,1).rvs((n,2))))
    """ Concatenation of repeating 0s and 1s """
    outcome = np.concatenate(( np.repeat(0,n), np.repeat(1,n) ))
    """ Plotting points for the first n and second n sets """
    plt.plot(points[:n,0],points[:n,1],'bo')
    plt.plot(points[n:,0],points[n:,1],'ro')
    return points, outcome


""" Assigning points and outcomes """
points, outcome = gen_synth_data(50)


""" Plotting the point along with the rest """
plt.scatter(p[0],p[1], s = 50, c = 'g')


""" Distance function to calculate distance between considered point and 
    synthetic data points """
def distance(points, p):
    dist = []
    for i in range(len(points)):
        """ Calculating the distances and appending to the list dist """
        dist.append(np.round(np.sqrt( np.power( (p[0]-points[i,0]), 2) \
        + np.power( (p[1]-points[i,1]), 2) ), 3))
    return dist


""" Distance passed as array """
dist = distance(points,p)


""" Passes array of index using one argument and after sorting in ascending order """        
ind = np.argsort(dist)


""" Nearest neighbours """
nn = []
out = []


""" Arranging the points and outcomes according to least distance """
for i in ind:
    nn.append(list(points[i]))
    out.append(outcome[i])


""" Verifying once through print statements """
print(nn[:5])
print()
print(out[:10])


""" Classifying our point on the basis of known values """    
def knn_classifier(out,k=5):
    
    """ Returns the value having max frequency as well as the value of the
       frequency """
    vote, count = ss.mstats.mode(out[:k])
    return vote

vote = int(knn_classifier(out,10))


""" Printing the classification result """
print()
print('The point '+str(tuple(p))+' belongs to the class: '+str(vote))
    
