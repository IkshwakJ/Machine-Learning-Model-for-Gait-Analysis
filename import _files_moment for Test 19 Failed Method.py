import os
import glob 
import pandas as pd
import numpy as np
from itertools import chain
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

Tot_file_data = []
file_labels_overall = []
for p in range(17):
    id = 'P' + str(p + 1) + 'Data'
    path = 'C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/' + id + '/Joints_Kinetics/*.sto'
    fileNames = glob.glob(path)
    file_labels_overall = []
    for fileName in fileNames:
        # print(fileName)
        file_header, file_labels, file_data = readMotionFile(filename = fileName)
        # os.system('cls||clear')
        # print(file_data)
        data = np.array(file_data)
        # print(data.shape)
        num_rows, num_cols = data.shape
        # print(file_labels)
        for i in range(num_rows):
            # print(data[i])
            if(i > 1):
                Tot_file_data.append(data[i])
        # print(np.asarray(Tot_file_data).shape)
        file_labels_overall = file_labels
    # print(Tot_file_data)
    # print(file_labels_overall)
df_moment = pd.DataFrame(Tot_file_data)
# print(df_moment.shape)
df_moment.columns = file_labels_overall
drop_list = ["time","knee_angle_r_moment","ankle_angle_r_moment","knee_angle_l_moment","ankle_angle_l_moment"]
df_moment = df_moment.drop(df_moment.columns.difference(drop_list), axis=1)
print(df_moment.describe())
# print(df_moment)

Tot_file_data = []
for p in range(17):
    id = 'P' + str(p + 1) + 'Data'
    path = 'C:/Users/ikshw/OneDrive/Desktop/ME 800/Joint Angles and Torques/Python/Data/' + id + '/Joints_Kinematics/*.mot'
    fileNames = glob.glob(path)
    file_labels_overall = []
    for fileName in fileNames:
        print(fileName)
        file_header, file_labels, file_data = readMotionFile(filename = fileName)
        # os.system('cls||clear')
        # print(file_data)
        data = np.array(file_data)
        num_rows, num_cols = data.shape
        temp_1 = []
        temp_2 = []
        for i in range(num_rows):
            if(i == 0):
                temp_1 = data[i]
            elif(i == 1):
                temp_2 = data[i]
            else:
                row_full = zip(temp_1, temp_2, data[i])
                # print(list(row_full))
                row_full = list(chain.from_iterable(row_full))
                Tot_file_data.append(row_full)
                temp_1 = temp_2
                temp_2 = data[i]
        # print(np.asarray(Tot_file_data).shape)
        file_labels_overall = file_labels
# print(file_labels_overall)
df_angles = pd.DataFrame(Tot_file_data)
Added_entry_labels = [x + '_prev_prev' for x in file_labels_overall]
Added_entry_labels_2 = [x + '_prev' for x in file_labels_overall]
file_labels_overall = zip(Added_entry_labels, Added_entry_labels_2, file_labels_overall)
file_labels_overall = list(chain.from_iterable(file_labels_overall))
# print(file_labels_overall)
df_angles.columns = file_labels_overall
# print(df_angles.describe())
drop_list = ["time_prev_prev","time_prev",
             "knee_angle_r_prev_prev","knee_angle_r_prev", "knee_angle_r",
             "ankle_angle_r_prev_prev", "ankle_angle_r_prev","ankle_angle_r",
             "knee_angle_l_prev_prev","knee_angle_l_prev","knee_angle_l",
             "ankle_angle_l_prev_prev","ankle_angle_l_prev","ankle_angle_l"]
df_angles = df_angles.drop(df_angles.columns.difference(drop_list), axis=1)
# print(df_angles.describe())
# print(df_angles)

horizontal_stack = pd.concat([df_moment, df_angles], axis=1)
print(horizontal_stack.describe())

horizontal_stack.to_csv("Moment_data_for_test_model_19.csv", sep=',')