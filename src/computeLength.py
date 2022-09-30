'''
length step estimation
'''

import numpy as np
import matplotlib.pyplot as plt
import computeSteps as cSteps
import peakAccelThreshold as pat

DATA_PATH = './data/'
GRAPH_PATH = './graphs/'

def get_data(data, timestamps, start, end):
    '''
    get selected data
    
    Parameters
    ----------
    data: array
        signal full of peaks
        
    timestamps : array
        Array of timestamps
        
    start: float
        start time
    
    end: float
        end time
    
    Returns
    -------
    array
        the conresponded data
        
    '''
    index_start = 0
    index_end = len(timestamps)-1
    while timestamps[index_start] < start:
        index_start += 1
    while timestamps[index_end] > end:
        index_end -= 1
    return data[index_start:index_end+1]
    
    
def weinberg_estimation(data, cst):
    '''
    Weinberg estimation method

    Parameters
    ----------
    data : array
        Signal full of peaks (1 step)
    
    Returns
    -------
    float
        The length of the step
        
    '''
    Amax = np.max(data)
    Amin = np.min(data)
    return cst * (Amax - Amin)**(1/4)

def kim_estimation(data, cst):
    '''
    Kim estimation method

    Parameters
    ----------
    data : array
        Signal full of peaks (1 step)
    
    Returns
    -------
    float
        The length of the step
        
    '''
    m = len(data)
    sum_acceleration = data.sum()
    print(sum_acceleration)
    
    return cst * (sum_acceleration/m)**(1/3)
    
                        
                                   
def computeStepLength(data, timestamps, cst):
    '''
    1st method

    Calculating steps length using weinberg estimation method
    
    Parameters
    ----------
    data : array
        Signal full of peaks
    timestamps : array
        Array of timestamps
    cst : float
        The threshold to use to filter peaks

    Returns
    -------
    array
        Array of steps length

    '''
    crossings = pat.peak_accel_threshold(data, timestamps, cst)
    steps = len(crossings)//2
    
    weinberg_length_list = np.empty(steps)
    #kim_length_list = np.empty(steps)
    
    for i in range(steps):
        start = crossings[i][0]
        end = crossings[i+2][0]
        weinberg_length_list[i] = weinberg_estimation(get_data(data,timestamps,start,end), 0.48)
        #kim_length_list[i] = kim_estimation(get_data(data,timestamps,start,end), 0.48)
    
    return weinberg_length_list


def plot_step_length(data):
    steps = len(data)
    distance = data.sum()
    plt.title("step length estimation: {} steps, {} m".format(steps,round(distance,2)))
    plt.xlabel("steps")
    plt.ylabel("length [m]")
    plt.bar(range(1,steps+1),data)
    plt.savefig(GRAPH_PATH+'steps_length')
    plt.show()
    
    
if __name__ == "__main__":
    
    #filter requirements
    order = 4
    fs = 100 
    cutoff = 2
    
    x_data, y_data, z_data, r_data, timestamps = cSteps.pull_data(DATA_PATH, 'Accelerometer')

    #filter
    r = cSteps.lowpass_filter(r_data, cutoff, fs, order)
    
    #withoutmean
    r = cSteps.data_without_mean(r)
    ZERO = 0
    
    weinberg_length_list = computeStepLength(r, timestamps, ZERO)
    
    plot_step_length(weinberg_length_list)
    