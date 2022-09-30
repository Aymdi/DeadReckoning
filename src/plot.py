'''
display sensors data using graphs
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import csv

DATA_PATH = './data/'
GRAPH_PATH = './graphs/'

def get_data_magnetometer(file_path):
    '''
    Gets time, x, y, z measurements for EMF
    input: file_path (string): relative path to the written csv file
    output: data (list): list of 4 tables 
    '''
    data=[[],[],[],[]]
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        line_counter = 0
        for row in reader:
            if line_counter > 0:
                for i in range(4):
                    data[i].append(float(row[i]))
            line_counter += 1
    return data

def get_data_accelerometer(file_path):
    '''
    Gets time, x, y, z measurements for acceleromter
    input: file_path (string): relative path to the written csv file
    output: data (list): list of 4 tables 
    '''
    data=[[],[],[],[]]
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        line_counter = 0
        for row in reader:
            if line_counter > 0:
                for i in range(4):
                    data[i].append(float(row[i]))
            line_counter += 1
    return data

def get_data_gyroscope(file_path):
    '''
    Gets time, x, y, z measurements for gyroscope
    input: file_path (string): relative path to the written csv file
    output: data (list): list of 4 tables 
    '''
    data=[[],[],[],[]]
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        line_counter = 0
        for row in reader:
            if line_counter > 0:
                for i in range(4):
                    data[i].append(float(row[i]))
            line_counter += 1
    return data

def get_data_location(file_path):
    '''
    Gets data values for location
    input: file_path (string): relative path to the written csv file
    output: data (list): list of 8 tables 
    '''
    data=[[],[],[],[],[],[],[],[]]
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        line_counter = 0
        for row in reader:
            if line_counter > 0:
                for i in range(8):
                    data[i].append(float(row[i]))
            line_counter += 1
    return data

def make_graph_magnetometer(file_path):
    '''
    Draws graph from csv file for magnetometer measurements
    input: file_path (string): relative path to the written csv file
    '''
    data = get_data_magnetometer(file_path)
    plt.plot(data[0], data[1], label=" X-axis")
    plt.plot(data[0], data[2], label=" Y-axis")
    plt.plot(data[0], data[3], label=" Z-axis")
    plt.title("Magnetometer measurments")
    plt.xlabel("time (s)")
    plt.ylabel("magnetic field (µT)")
    plt.grid(True, which="both", linestyle='--')
    plt.legend()

    plt.savefig(GRAPH_PATH+"Magnetometer")
    #plt.close()
    plt.show()

def make_graph_accelerometer(file_path):
    '''
    Draws graph from csv file for magnetometer measurements
    input: file_path (string): relative path to the written csv file
    '''
    data = get_data_accelerometer(file_path)
    fig, (x, y, z) = plt.subplots(3)
    fig.suptitle("Accelerometer measurements (m/s^2)")
    x.plot(data[0], data[1], label=" X-axis")
    plt.xlabel("time (s)")
    x.grid(True, which="both", linestyle='--')
    x.legend()
    y.plot(data[0], data[2], label=" Y-axis", color="orange")
    plt.xlabel("time (s)")
    y.grid(True, which="both", linestyle='--')
    y.legend()
    z.plot(data[0], data[3], label=" Z-axis", color="green")
    plt.xlabel("time (s)")
    z.grid(True, which="both", linestyle='--')
    z.legend()
    
    plt.savefig(GRAPH_PATH+"Accelerometer")
    #plt.close()
    plt.show()

def make_graph_accelerometer_magnetude(file_path):
    '''
    Draws graph from csv file for magnetometer measurements
    input: file_path (string): relative path to the written csv file
    '''
    data = get_data_accelerometer(file_path)
    r = []
    for i in range(len(data[1])):
        r.append(math.sqrt(data[1][i]**2 + data[2][i]**2 + data[3][i]**2))
    plt.plot(data[0], r)
    plt.title("Acceleration Norm")
    plt.savefig(GRAPH_PATH+"Acceleration Norm")

    plt.show()


def make_graph_gyroscope(file_path):
    '''
    Draws graph from csv file for magnetometer measurements
    input: file_path (string): relative path to the written csv file
    '''
    data = get_data_gyroscope(file_path)
    fig, (x, y, z) = plt.subplots(3)
    fig.suptitle("Gyroscope measurments (rad/s)")
    x.plot(data[0], data[1], label=" X-axis")
    plt.xlabel("time (s)")
    x.grid(True, which="both", linestyle='--')
    x.legend()
    y.plot(data[0], data[2], label=" Y-axis", color="orange")
    plt.xlabel("time (s)")
    y.grid(True, which="both", linestyle='--')
    y.legend()
    z.plot(data[0], data[3], label=" Z-axis", color="green")
    plt.xlabel("time (s)")
    z.grid(True, which="both", linestyle='--')
    z.legend()

    plt.savefig(GRAPH_PATH+"Gyroscope")
    #plt.close()
    plt.show()

def make_graph_location(file_path):
    '''
    Draws graph from csv file for location measurements
    input: file_path (string): relative path to the written csv file
    '''
    data = get_data_location(file_path)
    labels = ["Latitude (°)", "Longitude (°)", "Height (m)", "Velocity (m/s)", "Direction (°)", "Horizental Accuracy (m)", "Vertical Accuracy (°)"]
    fig, axes = plt.subplots(7)
    fig.suptitle("Location measurments")
    for i in range(7):
        if i>=5:
            axes[i].set_yscale('log')
        axes[i].plot(data[0], data[i+1], label = labels[i])
        plt.xlabel("time (s)")
        axes[i].grid(True, which="both", linestyle='--')
        axes[i].legend()

    #plt.savefig(file_path.split(".")[0]+'.png')
    #plt.close()
    plt.show()

def make_graph_location_onepertime(file_path):
    '''
    Draws graph from csv file for location measurements
    input: file_path (string): relative path to the written csv file
    '''
    data = get_data_location(file_path)
    labels = ["Latitude (°)", "Longitude (°)", "Height (m)", "Velocity (m/s)", "Direction (°)", "Horizental Accuracy (m)", "Vertical Accuracy (°)"]
    plt.title("Location measurments")

    for i in range(len(labels)):

        if i>=5:
            #plt.yscale("log")
            pass

        plt.plot(data[0], data[i], label = labels[i])
        plt.ylabel(labels[i])
        plt.xlabel("time (s)")
        plt.grid(True, which="both", linestyle='--')
        plt.legend()

        #plt.savefig(file_path.split(".")[0]+'.png')
        #plt.close()
        plt.show()


def make_graphs():
    
    #plt.style.use("ggplot")
    make_graph_magnetometer(DATA_PATH+'Magnetometer.csv')
    make_graph_accelerometer(DATA_PATH+"Accelerometer.csv")
    make_graph_accelerometer_magnetude(DATA_PATH+"Accelerometer.csv")
    make_graph_gyroscope(DATA_PATH+"Gyroscope.csv")
    #make_graph_location(DATA_PATH+"Location.csv")
    #make_graph_location_onepertime(DATA_PATH+"Location.csv")
    

if __name__ == "__main__":

    make_graphs()