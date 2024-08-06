import os
import glob 
import pandas as pd
import numpy as np
from itertools import chain
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
for j in range(1,18):
    if j==6 or j==8:
        continue
    p_id = j
    #Loop to extract entire data for one person 
    filenames_torque = glob.glob('C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/P' + str(p_id)+'Data/Joints_Kinetics/*.sto')
    filenames_position = glob.glob('C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/P' + str(p_id)+'Data/Joints_Kinematics/*.mot')
    id_num = 1

    for fileName in filenames_torque:
        fileName = 'C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/P' + str(p_id)+'Data/Joints_Kinetics/'+ str(id_num) + '.sto'
        Tot_file_data = []
        file_labels_overall = []
        horizontal_stack = []
        file_header, file_labels, file_data = readMotionFile(filename = fileName)
        data = np.array(file_data)
        # print(file_data)
        num_rows, num_cols = data.shape
        for i in range(num_rows):
            if(i>1):
                Tot_file_data.append(data[i])
        file_labels_overall = file_labels
        df_moment = pd.DataFrame(Tot_file_data)
        df_moment.columns = file_labels_overall
        drop_list = ["time","knee_angle_r_moment","ankle_angle_r_moment","knee_angle_l_moment","ankle_angle_l_moment"]
        df_moment = df_moment.drop(df_moment.columns.difference(drop_list), axis=1)
        # print(df_moment.describe)

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
        drop_list = ["time_prev_prev","time_prev", "time_next", 
                    "knee_angle_r_prev_prev","knee_angle_r_prev", "knee_angle_r", 
                    "ankle_angle_r_prev_prev", "ankle_angle_r_prev","ankle_angle_r", 
                    "knee_angle_l_prev_prev","knee_angle_l_prev","knee_angle_l", 
                    "ankle_angle_l_prev_prev","ankle_angle_l_prev","ankle_angle_l"]
        df_angles = df_angles.drop(df_angles.columns.difference(drop_list), axis=1)
        #print(df_angles.describe)

        horizontal_stack = pd.concat([df_moment, df_angles], axis=1)
        cols = ['time_prev_prev','time_prev','time',
                'knee_angle_r_moment','ankle_angle_r_moment','knee_angle_l_moment','ankle_angle_l_moment',
                'knee_angle_r_prev_prev','knee_angle_r_prev','knee_angle_r',
                'ankle_angle_r_prev_prev','ankle_angle_r_prev','ankle_angle_r',
                'knee_angle_l_prev_prev','knee_angle_l_prev','knee_angle_l',
                'ankle_angle_l_prev_prev','ankle_angle_l_prev','ankle_angle_l']
        horizontal_stack = horizontal_stack[cols]
        col_add = ' P' + str(p_id) + '_Run-' + str(id_num)
        cols = ['time_prev_prev','time_prev','time',
                'knee_angle_r_moment' + col_add,'ankle_angle_r_moment' + col_add,'knee_angle_l_moment' + col_add,'ankle_angle_l_moment' + col_add,
                'knee_angle_r_prev_prev' + col_add,'knee_angle_r_prev' + col_add,'knee_angle_r' + col_add,
                'ankle_angle_r_prev_prev' + col_add,'ankle_angle_r_prev' + col_add,'ankle_angle_r' + col_add,
                'knee_angle_l_prev_prev' + col_add,'knee_angle_l_prev' + col_add,'knee_angle_l' + col_add,
                'ankle_angle_l_prev_prev' + col_add,'ankle_angle_l_prev' + col_add,'ankle_angle_l' + col_add]
        horizontal_stack.columns = cols
        # print(horizontal_stack.describe)
        first_timestamp = horizontal_stack['time'][0]
        # if (id_num == 2):
        #     print(first_timestamp)
        #     print(horizontal_stack.time[0])
        #     print(horizontal_stack.time_prev[0])
        #     print(horizontal_stack.time_prev_prev[0])
        horizontal_stack['time'] -= first_timestamp
        horizontal_stack['time_prev'] -= first_timestamp
        horizontal_stack['time_prev_prev'] -= first_timestamp
        # print(id_num)
        # print(first_timestamp)
        # print(horizontal_stack.describe)
        horizontal_stack = horizontal_stack[cols]
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
    print(horizontal_stack_overall.describe)
    input("continue?: press enter")
horizontal_stack_overall.to_csv("Moment_data_overall.csv", sep=',')