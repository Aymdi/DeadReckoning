'''
Dead Reckoning using data sensors
'''

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import sin,cos,pi
import lowpass as lp
import statistics
import computeSteps as cACC

DATA_PATH = './data/'
GRAPH_PATH = './graphs/'

#filter requirements
order = 4
fs = 5000
cutoff = 2

def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    Hx = []
    Hy = []
    Hz = []
    He = []
    timestamps = []
    line_counter = 0
    for line in f:
        if line_counter > 0:
            value = line.split(',')
            if len(value) > 3:
                timestamps.append(float(value[0]))
                hx = float(value[1])
                hy = float(value[2])
                hz = float(value[3])
                r = math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
                Hx.append(hx)
                Hy.append(hy)
                Hz.append(hz)
                He.append(r)
        line_counter += 1
    return np.array(Hx), np.array(Hy), np.array(Hz), np.array(He), np.array(timestamps)

def data_corrected(data):
    'without mean'
    return np.array(data) - np.mean(data)

mag_data  = pd.read_csv(DATA_PATH+'Magnetometer.csv')
acc_data  = pd.read_csv(DATA_PATH+'Accelerometer.csv')
gyro_data = pd.read_csv(DATA_PATH+'Gyroscope.csv')
 
phone_mag = np.array([data_corrected(mag_data["X (µT)"]),
                      data_corrected(mag_data["Y (µT)"]),
                      data_corrected(mag_data["Z (µT)"])])

phone_acc = np.array([data_corrected(acc_data["X (m/s^2)"]),
                      data_corrected(acc_data["Y (m/s^2)"]),
                      data_corrected(acc_data["Z (m/s^2)"])])

pitch = gyro_data["X (rad/s)"]
roll  = gyro_data["Y (rad/s)"]
yaw   = gyro_data["Z (rad/s)"]

phone_mag_filtered = np.array([lp.butter_lowpass_filter(mag_data["X (µT)"],cutoff,fs,order),
                               lp.butter_lowpass_filter(mag_data["Y (µT)"],cutoff,fs,order),
                               lp.butter_lowpass_filter(mag_data["Z (µT)"],cutoff,fs,order)])

phone_acc_filtered = np.array([lp.butter_lowpass_filter(acc_data["X (m/s^2)"],cutoff,fs,order),
                               lp.butter_lowpass_filter(acc_data["Y (m/s^2)"],cutoff,fs,order),
                               lp.butter_lowpass_filter(acc_data["Z (m/s^2)"],cutoff,fs,order)])

pitch_filtered = lp.butter_lowpass_filter(gyro_data["X (rad/s)"],cutoff,fs,order)
roll_filtered  = lp.butter_lowpass_filter(gyro_data["Y (rad/s)"],cutoff,fs,order)
yaw_filtered   = lp.butter_lowpass_filter(gyro_data["Z (rad/s)"],cutoff,fs,order)


# Rotation matrices
def R_x(x):
    # body frame rotation about x axis
    return np.array([[1,      0,       0],
                     [0,cos(-x),-sin(-x)],
                     [0,sin(-x), cos(-x)]])

def R_y(y):
    # body frame rotation about y axis
    return np.array([[cos(-y),0,-sin(-y)],
                    [0,      1,        0],
                    [sin(-y), 0, cos(-y)]])

def R_z(z):
    # body frame rotation about z axis
    return np.array([[cos(-z),-sin(-z),0],
                     [sin(-z), cos(-z),0],
                     [0,      0,       1]])
    
# Init arrays for new transformed accelerations
earth_mag = np.empty(phone_mag.shape)

for i in range(mag_data.shape[0]):
    earth_mag[:,i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ phone_mag[:,i]

x_data, y_data, z_data = earth_mag[0], earth_mag[1], earth_mag[2]

timestamp = mag_data["Time (s)"]

def compute_direction(x, y): #atan2 
    res = 0
    if y>0:
        res = 90 - math.atan(x/y)*180/math.pi
    elif y<0:
        res = 270 - math.atan(x/y)*180/math.pi
    else:
        if x<0:
            res = 180
        else:
            res = 0
    return res

def compute_compass(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        compass.append(direction)
    return np.array(compass)

def compute_compass2(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        direction = (450-direction)*math.pi/180
        direction = 2*math.pi - direction
        compass.append(direction)
    return np.array(compass)

def compute_compass3(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = math.atan2(Hx[i],Hy[i])
        compass.append(direction)
    return np.array(compass)

def draw_arrows(x, y, angle, r=1, label=""):
    plt.arrow(x, y, r*math.sin(angle), r*math.cos(angle), head_width=.1, color="red",label=label)
    plt.arrow(x, y, 1, 0, head_width=.1, color="black")
    plt.annotate('N', xy=(x, y+r))
    plt.arrow(x, y, 0, 1, head_width=.1, color="black")
    plt.annotate('E', xy=(x+r, y))

def draw_all_arrows(data_x, data_y, data_angles, r=1):
    draw_arrows(data_x[0], data_y[0], data_angles[0], r, label="direction")
    for i in range(1,len(data_x)):
        draw_arrows(data_x[i], data_y[i], data_angles[i], r)

def compute_avg(data_x, data_y, steps):
    Hx_avg = []
    Hy_avg = []
    avg = int(len(data_x)/steps)
    x, y = 0, 0
    for i in range(len(data_x)):
        x += data_x[i]
        y += data_y[i]
        if (i+1)%avg == 0:
            Hx_avg.append(x/avg)
            Hy_avg.append(y/avg)
            x, y = 0, 0
    return np.array(Hx_avg), np.array(Hy_avg)
    
def make_graph_points():
    Hx = x_data
    Hy = y_data
    dataCompass = compute_compass(Hx,Hy)
    dataCompass2 = compute_compass2(Hx,Hy)
    dataAcc = pull_data(DATA_PATH,'Accelerometer')[3] #norme
    timestamps = mag_data["Time (s)"]
    plt.plot(timestamps, dataAcc, marker='.', label=" steps")
    draw_all_arrows(timestamps, dataAcc, dataCompass2, 0.1)
    plt.title("Compass Heading _ Heading-To-The-Car-Back-Pant-Pocket")
    plt.xlabel("time [s]")
    plt.ylabel("acceleration norm [m/s^2]")
    plt.grid(True, which="both", linestyle="")
    plt.legend()
    plt.show()

def make_graph_steps(nbr_steps, distance_traveled):
    dist_step_avg = distance_traveled / nbr_steps
    dist_steps = []
    steps_num = np.arange(1,nbr_steps+1)
    i = 0
    for i in range(1,int(nbr_steps)+1):
        dist_steps.append(dist_step_avg*i)
        i += 1
    dist_steps = np.array(dist_steps)
    plt.plot(steps_num, dist_steps,"o",color="blue", label="steps")
    Hx = x_data
    Hy = y_data
    Hx, Hy = compute_avg(Hx,Hy,nbr_steps)
    dataCompass = compute_compass(Hx,Hy)
    dataCompass2 = compute_compass2(Hx,Hy)
    draw_all_arrows(steps_num, dist_steps, dataCompass2, 1)
    plt.title("Compas heading per step")
    plt.xlabel("Step number")
    plt.ylabel("Distance traveled [m]")
    plt.grid(True, which="both", linestyle="-")
    plt.legend()
    plt.show()

def draw_arrows_target(x, y, angle, r=1, label="", display_direction=0):
    plt.arrow(x, y, dx =r*math.sin(angle), dy= r*math.cos(angle), head_width=.3, color="red",label=label)
    if display_direction == 1:
        plt.arrow(x, y, 1, 0, head_width=.1, color="black")
        plt.annotate('N', xy=(x, y+r))
        plt.arrow(x, y, 0, 1, head_width=.1, color="black")
        plt.annotate('E', xy=(x+r, y))
    return(x+r*math.sin(angle), y+r*math.cos(angle))

def draw_all_arrows_target(x0, y0, data_angles, r=1.5, display_steps=1, display_direction=0, display_stepsNumber=0):
    x,y = draw_arrows_target(x0, y0, data_angles[0], r, label="direction", display_direction=display_direction)
    plt.plot(x0,y0,".",color="blue", label="steps")
    plt.annotate('{}'.format(1), xy=(x0+0.3, y0), color="blue")
    for i in range(1, len(data_angles)):
        if display_steps == 1: plt.plot(x,y,".",color="blue")
        if display_stepsNumber == 1:  plt.annotate('{}'.format(i+1), xy=(x+0.3, y), color="blue")
        x, y = draw_arrows_target(x, y, data_angles[i], r, display_direction=display_direction)

def make_target(nbr_steps, distance_traveled, display_steps=1, display_direction=0, display_stepsNumber=0):
    Hx = earth_mag[0]
    Hy = earth_mag[1]
    Hx, Hy = compute_avg(Hx,Hy,nbr_steps)
    dataCompass = compute_compass(Hx,Hy)
    dataCompass2 = compute_compass2(Hx,Hy)
    draw_all_arrows_target(0,0,dataCompass2,1.5,display_steps,display_direction,display_stepsNumber)
    plt.legend()
    plt.savefig(GRAPH_PATH+'tracking')
    plt.show()


if __name__ == "__main__":
    
    steps = np.max(cACC.compute(display_graph = 0))
    dist = steps*0.69
    
    make_target(steps,
                dist,
                display_steps = 1,
                display_direction = 0,
                display_stepsNumber = 0)
    