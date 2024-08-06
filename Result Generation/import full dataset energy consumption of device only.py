import os
import glob 
import pandas as pd
import numpy as np
from numpy import absolute
from numpy import sin, cos, tan, radians
from itertools import chain
from math import sqrt, atan

threshold = 0.80 #This is the limit on the torque above which the system will provide the additional torque. It is in fraction.
motr_output_eff = 0.85 #This is the efficency of the motor in using electrical energy to provide output.
motor_regen_eff = 0.60 #This is the efficency of the motor in using kinetic energy to provide electrical output.
gear_train_eff = 0.85*0.95 #This is the efficiency of the geartrain that connects the motor to the joint.

hip_to_knee = 0.49 #in meters
knee_to_ankle = 0.36 #in meters
# These lenghts were taken from the average between the average lenghts for both genders from the link: https://multisite.eos.ncsu.edu/www-ergocenter-ncsu-edu/wp-content/uploads/sites/18/2016/06/Anthropometric-Detailed-Data-Tables.pdf
#moment of inertia in kg m^2 about a specified joint (using parallel axis theorem)
MoI_Knee_Upper_at_hip = 0.004669 + 1.778966*pow(0.021147 + hip_to_knee,2)
MoI_Knee_Lower_at_knee = 0.0000573746 + 0.073016*pow(0.114361 - 0.021147,2)
MoI_Ankle_Upper_at_knee = 0.004669 + 1.778966*pow(knee_to_ankle - (0.002676 + 0.033234),2)
MoI_Ankle_Lower_at_ankle = 0.001455 + 0.386672*pow(0.066936 - 0.033234,2)
def MoI_Knee_Lower_at_hip(knee_angle):
    joint_to_cm = sqrt(pow((0.114361 - 0.021147),2) + pow((0.027078 - 0.007908),2))
    dist_sqr = pow(hip_to_knee + cos(knee_angle*np.pi/180)*joint_to_cm,2) +  pow(sin(knee_angle*np.pi/180)*joint_to_cm,2)
    MoI_Knee_Lower_at_hip_value = 0.0000573746 + 0.073016*dist_sqr
    return MoI_Knee_Lower_at_hip_value

def MoI_Ankle_Upper_at_hip(knee_angle):
    knee_joint_to_cm = sqrt(pow((knee_to_ankle - 0.033234 - 0.002676),2) + pow((0.005403 - 0.005373),2))
    dist_sqr = pow(hip_to_knee + knee_joint_to_cm*cos(knee_angle*np.pi/180),2) +  pow(knee_joint_to_cm*sin(knee_angle*np.pi/180),2)
    MoI_Knee_Lower_at_hip_value = 0.004669 + 1.778996*dist_sqr
    return MoI_Knee_Lower_at_hip_value

def MoI_Ankle_Lower_at_knee(ankle_angle):
    ankle_joint_to_cm = sqrt(pow((0.066936 - 0.033234),2) + pow((0.021183 + 0.005403),2))
    dist_sqr = pow(knee_to_ankle - ankle_joint_to_cm*sin(ankle_angle*np.pi/180),2) +  pow(ankle_joint_to_cm*cos(ankle_angle*np.pi/180),2)
    MoI_Ankle_Lower_at_knee_value = 0.001455 + 0.386672*dist_sqr
    return MoI_Ankle_Lower_at_knee_value

def MoI_Ankle_Lower_at_hip(ankle_angle, knee_angle):
    ankle_joint_to_cm = sqrt(pow((0.066936 - 0.033234),2) + pow((0.021183 + 0.005403),2))
    theta_at_knee = abs(atan(ankle_joint_to_cm*cos(ankle_angle*np.pi/180)/knee_to_ankle - ankle_joint_to_cm*sin(ankle_angle*np.pi/180)))
    dist_at_knee = sqrt(pow(knee_to_ankle - ankle_joint_to_cm*sin(ankle_angle*np.pi/180),2) +  pow(ankle_joint_to_cm*cos(ankle_angle*np.pi/180),2))
    dist_sqr_at_hip = pow((dist_at_knee*cos(knee_angle*np.pi/180 - theta_at_knee) + hip_to_knee),2) + pow(dist_at_knee*sin(knee_angle*np.pi/180 - theta_at_knee),2)
    MoI_Ankle_Lower_at_knee_value = 0.001455 + 0.386672*dist_sqr_at_hip
    return MoI_Ankle_Lower_at_knee_value

def append_empty_rows(dataframe, n):
    for _ in range(n):
        dataframe.loc[len(dataframe)] = pd.Series(dtype='float64')

def readMotionFile(filename):
    """ Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])
        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()
    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data

horizontal_stack_overall = []
Total_Used_Power = 0
Total_Regen_Power = 0
count_runs = 0
for j in range(1,18):
    if j==6 or j==8:
        continue
    p_id = j
    #Loop to extract entire data for one person 
    filenames_torque = glob.glob('C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/P' + str(p_id)+'Data/Joints_Kinetics/*.sto')
    filenames_position = glob.glob('C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/P' + str(p_id)+'Data/Joints_Kinematics/*.mot')
    id_num = 1

    for fileName in filenames_torque:
        
        count_runs = count_runs + 1
        fileName = 'C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/P' + str(p_id)+'Data/Joints_Kinematics/'+ str(id_num) + '.mot'
        Tot_file_data = []
        file_labels_overall = []
        file_header, file_labels, file_data = readMotionFile(filename = fileName)
        data = []
        data = np.array(file_data)
        num_rows, num_cols = data.shape
        temp = []
        i = 0
        for i in range(num_rows):
            if(i == 0):
                temp_1 = data[i]
            elif(i == 1):
                temp_2 = data[i]
            else:
                row_full = zip(temp_1, temp_2, data[i])
                row_full = list(chain.from_iterable(row_full))
                Tot_file_data.append(row_full)
                temp_1 = temp_2
                temp_2 = data[i]
        file_labels_overall = file_labels
        df_angles = pd.DataFrame(Tot_file_data)

        Added_entry_labels = [x + '_prev_prev' for x in file_labels_overall]
        Added_entry_labels_2 = [x + '_prev' for x in file_labels_overall]
        file_labels_overall = zip(Added_entry_labels, Added_entry_labels_2, file_labels_overall)
        file_labels_overall = list(chain.from_iterable(file_labels_overall))
        df_angles.columns = file_labels_overall
        drop_list = ["time_prev_prev","time_prev","time", 
                    "hip_flexion_r_prev_prev","hip_flexion_r_prev","hip_flexion_r",
                    "knee_angle_r_prev_prev","knee_angle_r_prev", "knee_angle_r", 
                    "ankle_angle_r_prev_prev", "ankle_angle_r_prev","ankle_angle_r",
                    "hip_flexion_l_prev_prev","hip_flexion_l_prev","hip_flexion_l", 
                    "knee_angle_l_prev_prev","knee_angle_l_prev","knee_angle_l", 
                    "ankle_angle_l_prev_prev","ankle_angle_l_prev","ankle_angle_l"]
        df_angles = df_angles.drop(df_angles.columns.difference(drop_list), axis=1)
        # print(df_angles.describe)

        horizontal_stack = df_angles
        col_add = ' P' + str(p_id) + '_Run-' + str(id_num)
        cols = ['time_prev_prev','time_prev','time',
                'hip_flexion_r_prev_prev'+col_add,'hip_flexion_r_prev'+col_add,'hip_flexion_r'+col_add,
                'knee_angle_r_prev_prev' + col_add,'knee_angle_r_prev' + col_add,'knee_angle_r' + col_add,
                'ankle_angle_r_prev_prev' + col_add,'ankle_angle_r_prev' + col_add,'ankle_angle_r' + col_add,
                'hip_flexion_l_prev_prev'+col_add,'hip_flexion_l_prev'+col_add,'hip_flexion_l'+col_add,
                'knee_angle_l_prev_prev' + col_add,'knee_angle_l_prev' + col_add,'knee_angle_l' + col_add,
                'ankle_angle_l_prev_prev' + col_add,'ankle_angle_l_prev' + col_add,'ankle_angle_l' + col_add]
        horizontal_stack.columns = cols

        first_timestamp = horizontal_stack['time'][0]
        horizontal_stack['time'] -= first_timestamp
        horizontal_stack['time_prev'] -= first_timestamp
        horizontal_stack['time_prev_prev'] -= first_timestamp

        horizontal_stack['hip_r_ang_vel'+col_add] = 0.0174533*(horizontal_stack['hip_flexion_r'+col_add] - horizontal_stack['hip_flexion_r_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_r_ang_vel'+col_add] = 0.0174533*(horizontal_stack['knee_angle_r'+col_add] - horizontal_stack['knee_angle_r_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_ang_vel'+col_add] = 0.0174533*(horizontal_stack['ankle_angle_r'+col_add] - horizontal_stack['ankle_angle_r_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['hip_l_ang_vel'+col_add] = 0.0174533*(horizontal_stack['hip_flexion_l'+col_add] - horizontal_stack['hip_flexion_l_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_ang_vel'+col_add] = 0.0174533*(horizontal_stack['knee_angle_l'+col_add] - horizontal_stack['knee_angle_l_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_ang_vel'+col_add] = 0.0174533*(horizontal_stack['ankle_angle_l'+col_add] - horizontal_stack['ankle_angle_l_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])

        horizontal_stack['Energy_Device_Inst_Hip_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Inst_Hip_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Inst_Knee_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Inst_Knee_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add] = pd.Series(dtype='float')

        for q in range (horizontal_stack.shape[0]):
            if q == 0:
                horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q] = 0
                horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q] = 0
                horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q] = 0
                horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q] = 0
                horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q] = 0
                horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q] = 0    
            else:
                horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q] = 0.5*(
                                                                          MoI_Knee_Lower_at_hip(horizontal_stack['knee_angle_r'+ col_add][q])*pow(horizontal_stack['hip_r_ang_vel'+ col_add][q],2)+
                                                                          MoI_Knee_Upper_at_hip*pow(horizontal_stack['hip_r_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Lower_at_hip(horizontal_stack['ankle_angle_r'+col_add][q], horizontal_stack['knee_angle_r'+ col_add][q])*pow(horizontal_stack['hip_r_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Upper_at_hip(horizontal_stack['knee_angle_r'+ col_add][q])*pow(horizontal_stack['hip_r_ang_vel'+ col_add][q],2)
                                                                          ) - horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q-1]
                horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q] = 0.5*(
                                                                          MoI_Knee_Lower_at_hip(horizontal_stack['knee_angle_l'+ col_add][q])*pow(horizontal_stack['hip_l_ang_vel'+ col_add][q],2)+
                                                                          MoI_Knee_Upper_at_hip*pow(horizontal_stack['hip_l_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Lower_at_hip(horizontal_stack['ankle_angle_l'+col_add][q], horizontal_stack['knee_angle_l'+ col_add][q])*pow(horizontal_stack['hip_l_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Upper_at_hip(horizontal_stack['knee_angle_l'+ col_add][q])*pow(horizontal_stack['hip_l_ang_vel'+ col_add][q],2)
                                                                          ) - horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q-1]
                horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q] = 0.5*(MoI_Knee_Lower_at_knee*pow(horizontal_stack['knee_r_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Lower_at_knee(horizontal_stack['ankle_angle_r' + col_add][q])*pow(horizontal_stack['knee_r_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Upper_at_knee*pow(horizontal_stack['knee_r_ang_vel'+ col_add][q],2)
                                                                         ) - horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q-1]
                horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q] = 0.5*(MoI_Knee_Lower_at_knee*pow(horizontal_stack['knee_l_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Lower_at_knee(horizontal_stack['ankle_angle_l' + col_add][q])*pow(horizontal_stack['knee_l_ang_vel'+ col_add][q],2)+
                                                                          MoI_Ankle_Upper_at_knee*pow(horizontal_stack['knee_l_ang_vel'+ col_add][q],2)
                                                                         ) - horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q-1]
                horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q] = 0.5*MoI_Ankle_Lower_at_ankle*pow(horizontal_stack['ankle_r_ang_vel'+col_add][q],2) - horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q-1]
                horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q] = 0.5*MoI_Ankle_Lower_at_ankle*pow(horizontal_stack['ankle_l_ang_vel'+col_add][q],2) - horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q-1]
        
        horizontal_stack['Energy_Device_Used_Hip_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Used_Hip_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Used_Knee_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Used_Knee_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Used_Ankle_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Used_Ankle_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Used_Total'+col_add] = pd.Series(dtype='float')
        
        horizontal_stack['Energy_Device_Regen_Hip_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Regen_Hip_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Regen_Knee_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Regen_Knee_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Device_Regen_Total'+col_add] = pd.Series(dtype='float')

        for p in range (horizontal_stack.shape[0]):
            if p ==0:
                horizontal_stack['Energy_Device_Used_Hip_R'+col_add][p] = 0
                horizontal_stack['Energy_Device_Used_Hip_L'+col_add][p] = 0
                horizontal_stack['Energy_Device_Used_Knee_R'+col_add][p] = 0
                horizontal_stack['Energy_Device_Used_Knee_L'+col_add][p] = 0
                horizontal_stack['Energy_Device_Used_Ankle_R'+col_add][p] = 0
                horizontal_stack['Energy_Device_Used_Ankle_L'+col_add][p] = 0
                horizontal_stack['Energy_Device_Used_Total'+col_add][p] = 0

                horizontal_stack['Energy_Device_Regen_Hip_R'+col_add][p] = 0
                horizontal_stack['Energy_Device_Regen_Hip_L'+col_add][p] = 0
                horizontal_stack['Energy_Device_Regen_Knee_R'+col_add][p] = 0
                horizontal_stack['Energy_Device_Regen_Knee_L'+col_add][p] = 0
                horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add][p] = 0
                horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add][p] = 0
                horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = 0
            else:
                # if horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][p] >=0:
                #     horizontal_stack['Energy_Device_Used_Hip_R'+col_add][p] = horizontal_stack['Energy_Device_Used_Hip_R'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q]
                #     horizontal_stack['Energy_Device_Regen_Hip_R'+col_add][p] = horizontal_stack['Energy_Device_Regen_Hip_R'+col_add][p-1]
                #     horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q]
                #     horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1]
                # else:
                #     horizontal_stack['Energy_Device_Regen_Hip_R'+col_add][p] = horizontal_stack['Energy_Device_Regen_Hip_R'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q]
                #     horizontal_stack['Energy_Device_Used_Hip_R'+col_add][p] = horizontal_stack['Energy_Device_Used_Hip_R'+col_add][p-1]
                #     horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] 
                #     horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Hip_R'+col_add][q]

                # if horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][p] >=0:
                #     horizontal_stack['Energy_Device_Used_Hip_L'+col_add][p] = horizontal_stack['Energy_Device_Used_Hip_L'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q]
                #     horizontal_stack['Energy_Device_Regen_Hip_L'+col_add][p] = horizontal_stack['Energy_Device_Regen_Hip_L'+col_add][p-1]
                #     horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q]
                #     horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1]
                # else:
                #     horizontal_stack['Energy_Device_Regen_Hip_L'+col_add][p] = horizontal_stack['Energy_Device_Regen_Hip_L'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q]
                #     horizontal_stack['Energy_Device_Used_Hip_L'+col_add][p] = horizontal_stack['Energy_Device_Used_Hip_L'+col_add][p-1]
                #     horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] 
                #     horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Hip_L'+col_add][q]

                if horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][p] >=0:
                    horizontal_stack['Energy_Device_Used_Knee_R'+col_add][p] = horizontal_stack['Energy_Device_Used_Knee_R'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Knee_R'+col_add][p] = horizontal_stack['Energy_Device_Regen_Knee_R'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1]
                else:
                    horizontal_stack['Energy_Device_Regen_Knee_R'+col_add][p] = horizontal_stack['Energy_Device_Regen_Knee_R'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q]
                    horizontal_stack['Energy_Device_Used_Knee_R'+col_add][p] = horizontal_stack['Energy_Device_Used_Knee_R'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] 
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Knee_R'+col_add][q]

                if horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][p] >=0:
                    horizontal_stack['Energy_Device_Used_Knee_L'+col_add][p] = horizontal_stack['Energy_Device_Used_Knee_L'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Knee_L'+col_add][p] = horizontal_stack['Energy_Device_Regen_Knee_L'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1]
                else:
                    horizontal_stack['Energy_Device_Regen_Knee_L'+col_add][p] = horizontal_stack['Energy_Device_Regen_Knee_L'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q]
                    horizontal_stack['Energy_Device_Used_Knee_L'+col_add][p] = horizontal_stack['Energy_Device_Used_Knee_L'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] 
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Knee_L'+col_add][q]

                if horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][p] >=0:
                    horizontal_stack['Energy_Device_Used_Ankle_R'+col_add][p] = horizontal_stack['Energy_Device_Used_Ankle_R'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add][p] = horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1]
                else:
                    horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add][p] = horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q]
                    horizontal_stack['Energy_Device_Used_Ankle_R'+col_add][p] = horizontal_stack['Energy_Device_Used_Ankle_R'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] 
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add][q]

                if horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][p] >=0:
                    horizontal_stack['Energy_Device_Used_Ankle_L'+col_add][p] = horizontal_stack['Energy_Device_Used_Ankle_L'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add][p] = horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] + horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q]
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1]
                else:
                    horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add][p] = horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q]
                    horizontal_stack['Energy_Device_Used_Ankle_L'+col_add][p] = horizontal_stack['Energy_Device_Used_Ankle_L'+col_add][p-1]
                    horizontal_stack['Energy_Device_Used_Total'+col_add][p] = horizontal_stack['Energy_Device_Used_Total'+col_add][p-1] 
                    horizontal_stack['Energy_Device_Regen_Total'+col_add][p] = horizontal_stack['Energy_Device_Regen_Total'+col_add][p-1] - horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add][q]
    
        drop_list = ['time_prev_prev','time_prev','time', 
                    # 'Energy_Device_Inst_Hip_R'+col_add, 'Energy_Device_Inst_Hip_L'+col_add, 'Energy_Device_Used_Hip_R'+col_add,
                    # 'Energy_Device_Used_Hip_L'+col_add, 'Energy_Device_Regen_Hip_R'+col_add, 'Energy_Device_Regen_Hip_L'+col_add,
                    'Energy_Device_Inst_Knee_R'+col_add, 'Energy_Device_Inst_Knee_L'+col_add, 'Energy_Device_Used_Knee_R'+col_add,
                    'Energy_Device_Used_Knee_L'+col_add, 'Energy_Device_Regen_Knee_R'+col_add, 'Energy_Device_Regen_Knee_L'+col_add,
                    'Energy_Device_Inst_Ankle_R'+col_add, 'Energy_Device_Inst_Ankle_L'+col_add, 'Energy_Device_Used_Ankle_R'+col_add,
                    'Energy_Device_Used_Ankle_L'+col_add, 'Energy_Device_Regen_Ankle_R'+col_add, 'Energy_Device_Regen_Ankle_L'+col_add,
                    'Energy_Device_Used_Total'+col_add, 'Energy_Device_Regen_Total'+col_add]            
        horizontal_stack = horizontal_stack.drop(horizontal_stack.columns.difference(drop_list), axis=1)
        
        # horizontal_stack['Energy_Device_Inst_Hip_R'+col_add] = horizontal_stack['Energy_Device_Inst_Hip_R'+col_add]/motr_output_eff
        # horizontal_stack['Energy_Device_Inst_Hip_L'+col_add] = horizontal_stack['Energy_Device_Inst_Hip_L'+col_add]/motr_output_eff
        # horizontal_stack['Energy_Device_Used_Hip_R'+col_add] = horizontal_stack['Energy_Device_Used_Hip_R'+col_add]/motr_output_eff
        # horizontal_stack['Energy_Device_Used_Hip_L'+col_add] = horizontal_stack['Energy_Device_Used_Hip_L'+col_add]/motr_output_eff
        # horizontal_stack['Energy_Device_Regen_Hip_R'+col_add] = horizontal_stack['Energy_Device_Regen_Hip_R'+col_add]*motor_regen_eff*gear_train_eff
        # horizontal_stack['Energy_Device_Regen_Hip_L'+col_add] = horizontal_stack['Energy_Device_Regen_Hip_L'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['Energy_Device_Inst_Knee_R'+col_add] = horizontal_stack['Energy_Device_Inst_Knee_R'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Inst_Knee_L'+col_add] = horizontal_stack['Energy_Device_Inst_Knee_L'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Used_Knee_R'+col_add] = horizontal_stack['Energy_Device_Used_Knee_R'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Used_Knee_L'+col_add] = horizontal_stack['Energy_Device_Used_Knee_L'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Regen_Knee_R'+col_add] = horizontal_stack['Energy_Device_Regen_Knee_R'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['Energy_Device_Regen_Knee_L'+col_add] = horizontal_stack['Energy_Device_Regen_Knee_L'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add] = horizontal_stack['Energy_Device_Inst_Ankle_R'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add] = horizontal_stack['Energy_Device_Inst_Ankle_L'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Used_Ankle_R'+col_add] = horizontal_stack['Energy_Device_Used_Ankle_R'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Used_Ankle_L'+col_add] = horizontal_stack['Energy_Device_Used_Ankle_L'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add] = horizontal_stack['Energy_Device_Regen_Ankle_R'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add] = horizontal_stack['Energy_Device_Regen_Ankle_L'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['Energy_Device_Used_Total'+col_add] = horizontal_stack['Energy_Device_Used_Total'+col_add]/motr_output_eff
        horizontal_stack['Energy_Device_Regen_Total'+col_add] = horizontal_stack['Energy_Device_Regen_Total'+col_add]*motor_regen_eff*gear_train_eff
        Total_Used_Power = Total_Used_Power + horizontal_stack['Energy_Device_Used_Total'+col_add][horizontal_stack.shape[0]-1]/horizontal_stack['time'][horizontal_stack.shape[0]-1]
        Total_Regen_Power = Total_Regen_Power + horizontal_stack['Energy_Device_Regen_Total'+col_add][horizontal_stack.shape[0]-1]/horizontal_stack['time'][horizontal_stack.shape[0]-1]

        if (id_num == 1 and p_id == 1):
            horizontal_stack_overall = horizontal_stack
        else:
            #checking to see if the number of rows match, if not add additional NaN rows to the one with less rows 
            if horizontal_stack.shape[0] > horizontal_stack_overall.shape[0]:
                append_empty_rows(horizontal_stack_overall, (horizontal_stack.shape[0]-horizontal_stack_overall.shape[0]))
                horizontal_stack_overall['time_prev_prev'] = horizontal_stack['time_prev_prev']
                horizontal_stack_overall['time_prev'] = horizontal_stack['time_prev']
                horizontal_stack_overall['time'] = horizontal_stack['time']
            elif horizontal_stack.shape[0] < horizontal_stack_overall.shape[0]:
                append_empty_rows(horizontal_stack, (horizontal_stack_overall.shape[0]-horizontal_stack.shape[0]))
            horizontal_stack = horizontal_stack.drop(['time_prev_prev','time_prev','time'],axis = 1)
            horizontal_stack_overall = pd.concat([horizontal_stack_overall,horizontal_stack], axis = 1)
        id_num = id_num + 1
    # print(horizontal_stack_overall.describe)
    # input("continue?: press enter")
horizontal_stack_overall.to_csv("Energy_and_Power_based_on_device_only(No Hip).csv", sep=',')
Total_Used_Power = Total_Used_Power/count_runs
Total_Regen_Power = Total_Regen_Power/count_runs
print("Total Used Power On Average: ", Total_Used_Power)
print("Total Regenerated Power On Average: ", Total_Regen_Power)
print("Total Count: ", count_runs)