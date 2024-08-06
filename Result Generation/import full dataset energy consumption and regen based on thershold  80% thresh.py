import os
import glob 
import pandas as pd
import numpy as np
from numpy import absolute
from numpy import sin, cos, tan, radians
from itertools import chain

threshold = 0.80 #This is the limit on the torque above which the system will provide the additional torque. It is in fraction.
motr_output_eff = 0.85 #This is the efficency of the motor in using electrical energy to provide output.
motor_regen_eff = 0.60 #This is the efficency of the motor in using kinetic energy to provide electrical output.
gear_train_eff = 0.85*0.95 #This is the efficiency of the geartrain that connects the motor to the joint.

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

# Coeffs from ML model to predict joint moments

# # Right Knee 
def moment_at_right_knee(t0, t1, t2, rk_p_p_ang, rk_p_ang, rk_ang, ra_p_p_ang, ra_p_ang, ra_ang, lk_p_p_ang, lk_p_ang, lk_ang, la_p_p_ang, la_p_ang, la_ang):
    Intercept = -1788961.187
    R_K_prev_angle = -0.240993901
    R_K_angle = -70751.26321
    R_A_prev_angle = 4.36822142
    R_A_angle = 497214.9787
    L_K_prev_angle = -2.855908563
    L_K_angle = 3463.645914
    L_A_prev_angle = 2.456319117
    L_A_angle = -9310.888986
    R_K_angular_vel = -0.032023438
    R_K_angular_acc = 0.000472486
    R_A_angular_vel = 0.056488463
    R_A_angular_acc = -0.0000782
    L_K_angular_vel = -0.040751067
    L_K_angular_acc = -0.000691402
    L_A_angular_vel = -0.012812366
    L_A_angular_acc = -0.0000404
    R_K_sin	= 4220714.299
    R_K_sin2 = -83324.41867
    R_K_cos	= 2315241.596
    R_K_cos2 = -131642.5833	
    R_A_sin	= -30432561.56
    R_A_sin2 = 900502.4137
    R_A_cos	= 120379.6983
    R_A_cos2 = -6969.205659
    L_K_sin	= -210373.5235
    L_K_sin2 = 6030.663869
    L_K_cos	= -71214.40847
    L_K_cos2 = 4874.239778
    L_A_sin	= 567778.0185
    L_A_sin2 = -11337.43009
    L_A_cos	= -472804.6905
    L_A_cos2 = 31064.89191
    R_K_angle_pow2 = 272.5339965
    R_K_angle_pow3 = 3.164818312
    R_A_angle_pow2 = 14.06340438
    R_A_angle_pow3 = -20.84127162
    L_K_angle_pow2 = -7.832301818
    L_K_angle_pow3 = -0.141789531
    L_A_angle_pow2 = -53.10883735
    L_A_angle_pow3 = 0.443619739
    R_K_angle_tan = -382.7850389
    R_A_angle_tan = 143051.1137
    L_K_angle_tan = 6.651707269
    L_A_angle_tan = -11800.35633
    R_K_moment = (Intercept +
                R_K_prev_angle*rk_p_ang + R_K_angle*rk_ang + R_A_prev_angle*ra_p_ang + R_A_angle*ra_ang +
                L_K_prev_angle*lk_p_ang + L_K_angle*lk_ang + L_A_prev_angle*la_p_ang + L_A_angle*la_ang +
                R_K_angular_vel*(rk_ang-rk_p_ang)/(t2-t1) + R_K_angular_acc*((rk_ang-rk_p_ang)/(t2-t1) - (rk_p_ang-rk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                R_A_angular_vel*(ra_ang-ra_p_ang)/(t2-t1) + R_A_angular_acc*((ra_ang-ra_p_ang)/(t2-t1) - (ra_p_ang-ra_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_K_angular_vel*(lk_ang-lk_p_ang)/(t2-t1) + L_K_angular_acc*((lk_ang-lk_p_ang)/(t2-t1) - (lk_p_ang-lk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_A_angular_vel*(la_ang-la_p_ang)/(t2-t1) + L_A_angular_acc*((la_ang-la_p_ang)/(t2-t1) - (la_p_ang-la_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) + 
                R_K_sin*sin(radians(rk_ang)) + R_K_sin2*sin(2*radians(rk_ang)) + R_K_cos*cos(radians(rk_ang)) + R_K_cos2*cos(2*radians(rk_ang)) + 
                R_A_sin*sin(radians(ra_ang)) + R_A_sin2*sin(2*radians(ra_ang)) + R_A_cos*cos(radians(ra_ang)) + R_A_cos2*cos(2*radians(ra_ang)) +
                L_K_sin*sin(radians(lk_ang)) + L_K_sin2*sin(2*radians(lk_ang)) + L_K_cos*cos(radians(lk_ang)) + L_K_cos2*cos(2*radians(lk_ang)) + 
                L_A_sin*sin(radians(la_ang)) + L_A_sin2*sin(2*radians(la_ang)) + L_A_cos*cos(radians(la_ang)) + L_A_cos2*cos(2*radians(la_ang)) +
                R_K_angle_pow2*rk_ang*rk_ang + R_K_angle_pow3*rk_ang*rk_ang*rk_ang + R_A_angle_pow2*ra_ang*ra_ang + R_A_angle_pow3*ra_ang*ra_ang*ra_ang +
                L_K_angle_pow2*lk_ang*lk_ang + L_K_angle_pow3*lk_ang*lk_ang*lk_ang + L_A_angle_pow2*la_ang*la_ang + L_A_angle_pow3*la_ang*la_ang*la_ang +
                R_K_angle_tan*tan(radians(rk_ang)) + R_A_angle_tan*tan(radians(ra_ang)) + L_K_angle_tan*tan(radians(lk_ang)) + L_A_angle_tan*tan(radians(la_ang))
                )
    return R_K_moment

# # Right Ankle 
def moment_at_right_ankle(t0, t1, t2, rk_p_p_ang, rk_p_ang, rk_ang, ra_p_p_ang, ra_p_ang, ra_ang, lk_p_p_ang, lk_p_ang, lk_ang, la_p_p_ang, la_p_ang, la_ang):
    Intercept = -2537635.289
    R_K_prev_angle = -4.038015134
    R_K_angle = -111959.0614
    R_A_prev_angle = 0.011454307
    R_A_angle = -29743.96483
    L_K_prev_angle = 3.465736439
    L_K_angle = -25306.15951
    L_A_prev_angle = 0.552615782
    L_A_angle = -927629.4824
    R_K_angular_vel = 0.036584544
    R_K_angular_acc = -0.000675423
    R_A_angular_vel = -0.003250564
    R_A_angular_acc = 0.000114636
    L_K_angular_vel = -0.056811918
    L_K_angular_acc = 0.001502039
    L_A_angular_vel = -0.015494804
    L_A_angular_acc = 0.000230615
    R_K_sin	= 6651527.327
    R_K_sin2 = -117993.0416
    R_K_cos	= 3961528.058
    R_K_cos2 = -220334.5184	
    R_A_sin	= 1838946.176
    R_A_sin2 = -40546.70027
    R_A_cos	= -1896657.235
    R_A_cos2 = 122231.3119
    L_K_sin	= 1505239.322
    L_K_sin2 = -27573.82316
    L_K_cos	= 855689.2666
    L_K_cos2 = -45868.84628
    L_A_sin	= 56722996.26
    L_A_sin2 = -1616159.033
    L_A_cos	= -255781.3859
    L_A_cos2 = 16793.08955
    R_K_angle_pow2 = 469.4587196
    R_K_angle_pow3 = 5.093416467
    R_A_angle_pow2 = -214.4701539
    R_A_angle_pow3 = 1.442777405
    L_K_angle_pow2 = 102.7378153
    L_K_angle_pow3 = 1.155564071
    L_A_angle_pow2 = -28.74117439
    L_A_angle_pow3 = 39.41221973
    R_K_angle_tan = -482.3616861
    R_A_angle_tan = -53794.23803
    L_K_angle_tan = -195.5291978
    L_A_angle_tan = -341385.2407
    R_A_moment = (Intercept +
                R_K_prev_angle*rk_p_ang + R_K_angle*rk_ang + R_A_prev_angle*ra_p_ang + R_A_angle*ra_ang +
                L_K_prev_angle*lk_p_ang + L_K_angle*lk_ang + L_A_prev_angle*la_p_ang + L_A_angle*la_ang +
                R_K_angular_vel*(rk_ang-rk_p_ang)/(t2-t1) + R_K_angular_acc*((rk_ang-rk_p_ang)/(t2-t1) - (rk_p_ang-rk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                R_A_angular_vel*(ra_ang-ra_p_ang)/(t2-t1) + R_A_angular_acc*((ra_ang-ra_p_ang)/(t2-t1) - (ra_p_ang-ra_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_K_angular_vel*(lk_ang-lk_p_ang)/(t2-t1) + L_K_angular_acc*((lk_ang-lk_p_ang)/(t2-t1) - (lk_p_ang-lk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_A_angular_vel*(la_ang-la_p_ang)/(t2-t1) + L_A_angular_acc*((la_ang-la_p_ang)/(t2-t1) - (la_p_ang-la_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) + 
                R_K_sin*sin(radians(rk_ang)) + R_K_sin2*sin(2*radians(rk_ang)) + R_K_cos*cos(radians(rk_ang)) + R_K_cos2*cos(2*radians(rk_ang)) + 
                R_A_sin*sin(radians(ra_ang)) + R_A_sin2*sin(2*radians(ra_ang)) + R_A_cos*cos(radians(ra_ang)) + R_A_cos2*cos(2*radians(ra_ang)) +
                L_K_sin*sin(radians(lk_ang)) + L_K_sin2*sin(2*radians(lk_ang)) + L_K_cos*cos(radians(lk_ang)) + L_K_cos2*cos(2*radians(lk_ang)) + 
                L_A_sin*sin(radians(la_ang)) + L_A_sin2*sin(2*radians(la_ang)) + L_A_cos*cos(radians(la_ang)) + L_A_cos2*cos(2*radians(la_ang)) +
                R_K_angle_pow2*rk_ang*rk_ang + R_K_angle_pow3*rk_ang*rk_ang*rk_ang + R_A_angle_pow2*ra_ang*ra_ang + R_A_angle_pow3*ra_ang*ra_ang*ra_ang +
                L_K_angle_pow2*lk_ang*lk_ang + L_K_angle_pow3*lk_ang*lk_ang*lk_ang + L_A_angle_pow2*la_ang*la_ang + L_A_angle_pow3*la_ang*la_ang*la_ang +
                R_K_angle_tan*tan(radians(rk_ang)) + R_A_angle_tan*tan(radians(ra_ang)) + L_K_angle_tan*tan(radians(lk_ang)) + L_A_angle_tan*tan(radians(la_ang))
                )
    return R_A_moment

# # Left Knee 
def moment_at_left_knee(t0, t1, t2, rk_p_p_ang, rk_p_ang, rk_ang, ra_p_p_ang, ra_p_ang, ra_ang, lk_p_p_ang, lk_p_ang, lk_ang, la_p_p_ang, la_p_ang, la_ang):
    Intercept = -1017679.052
    R_K_prev_angle = -2.364485852
    R_K_angle = -51566.96082
    R_A_prev_angle = -0.09094483
    R_A_angle = 16080.3292
    L_K_prev_angle = -1.091775759
    L_K_angle = -30863.44628
    L_A_prev_angle = -0.36751845
    L_A_angle = -10867.64189
    R_K_angular_vel = -0.039947264
    R_K_angular_acc = -0.000427867
    R_A_angular_vel = -0.023780894
    R_A_angular_acc = 0.0000708427522226884
    L_K_angular_vel = -0.030003789
    L_K_angular_acc = 0.000918594
    L_A_angular_vel = 0.04428518
    L_A_angular_acc = 0.0000371307853868074
    R_K_sin	= 3080476.738
    R_K_sin2 = -62723.52376
    R_K_cos	= 1615030.055
    R_K_cos2 = -91905.78484	
    R_A_sin	= -976044.28
    R_A_sin2 = 45613.60284
    R_A_cos	= -1515278.972
    R_A_cos2 = 97959.63545
    L_K_sin	= 1885538.936
    L_K_sin2 = -58518.76497
    L_K_cos	= 520106.1906
    L_K_cos2 = -38278.73007
    L_A_sin	= 673035.6111
    L_A_sin2 = -34947.96091
    L_A_cos	= 460289.9373
    L_A_cos2 = -30288.02921
    R_K_angle_pow2 = 190.0313453
    R_K_angle_pow3 = 2.295486599
    R_A_angle_pow2 = -171.1627398
    R_A_angle_pow3 = -0.474268983
    L_K_angle_pow2 = 55.50560602
    L_K_angle_pow3 = 1.238031166
    L_A_angle_pow2 = 51.68321812
    L_A_angle_pow3 = 0.314920212
    R_K_angle_tan = -405.4570064
    R_A_angle_tan = -36575.17414
    L_K_angle_tan = -366.620242
    L_A_angle_tan = 19557.64375
    L_K_moment = (Intercept +
                R_K_prev_angle*rk_p_ang + R_K_angle*rk_ang + R_A_prev_angle*ra_p_ang + R_A_angle*ra_ang +
                L_K_prev_angle*lk_p_ang + L_K_angle*lk_ang + L_A_prev_angle*la_p_ang + L_A_angle*la_ang +
                R_K_angular_vel*(rk_ang-rk_p_ang)/(t2-t1) + R_K_angular_acc*((rk_ang-rk_p_ang)/(t2-t1) - (rk_p_ang-rk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                R_A_angular_vel*(ra_ang-ra_p_ang)/(t2-t1) + R_A_angular_acc*((ra_ang-ra_p_ang)/(t2-t1) - (ra_p_ang-ra_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_K_angular_vel*(lk_ang-lk_p_ang)/(t2-t1) + L_K_angular_acc*((lk_ang-lk_p_ang)/(t2-t1) - (lk_p_ang-lk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_A_angular_vel*(la_ang-la_p_ang)/(t2-t1) + L_A_angular_acc*((la_ang-la_p_ang)/(t2-t1) - (la_p_ang-la_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) + 
                R_K_sin*sin(radians(rk_ang)) + R_K_sin2*sin(2*radians(rk_ang)) + R_K_cos*cos(radians(rk_ang)) + R_K_cos2*cos(2*radians(rk_ang)) + 
                R_A_sin*sin(radians(ra_ang)) + R_A_sin2*sin(2*radians(ra_ang)) + R_A_cos*cos(radians(ra_ang)) + R_A_cos2*cos(2*radians(ra_ang)) +
                L_K_sin*sin(radians(lk_ang)) + L_K_sin2*sin(2*radians(lk_ang)) + L_K_cos*cos(radians(lk_ang)) + L_K_cos2*cos(2*radians(lk_ang)) + 
                L_A_sin*sin(radians(la_ang)) + L_A_sin2*sin(2*radians(la_ang)) + L_A_cos*cos(radians(la_ang)) + L_A_cos2*cos(2*radians(la_ang)) +
                R_K_angle_pow2*rk_ang*rk_ang + R_K_angle_pow3*rk_ang*rk_ang*rk_ang + R_A_angle_pow2*ra_ang*ra_ang + R_A_angle_pow3*ra_ang*ra_ang*ra_ang +
                L_K_angle_pow2*lk_ang*lk_ang + L_K_angle_pow3*lk_ang*lk_ang*lk_ang + L_A_angle_pow2*la_ang*la_ang + L_A_angle_pow3*la_ang*la_ang*la_ang +
                R_K_angle_tan*tan(radians(rk_ang)) + R_A_angle_tan*tan(radians(ra_ang)) + L_K_angle_tan*tan(radians(lk_ang)) + L_A_angle_tan*tan(radians(la_ang))
                )
    return L_K_moment

# # Left Ankle 
def moment_at_left_ankle(t0, t1, t2, rk_p_p_ang, rk_p_ang, rk_ang, ra_p_p_ang, ra_p_ang, ra_ang, lk_p_p_ang, lk_p_ang, lk_ang, la_p_p_ang, la_p_ang, la_ang):
    Intercept = 843965.7447
    R_K_prev_angle = 1.451769379
    R_K_angle = -49447.6466
    R_A_prev_angle = 0.465814642
    R_A_angle = 1176745.196
    L_K_prev_angle = -4.539003251
    L_K_angle = 20642.19039
    L_A_prev_angle = -2.096885857
    L_A_angle = 72870.53864
    R_K_angular_vel = -0.061375657
    R_K_angular_acc = 0.000625897
    R_A_angular_vel = -0.024204644
    R_A_angular_acc = 0.00000107741
    L_K_angular_vel = 0.061061877
    L_K_angular_acc = -0.001287437
    L_A_angular_vel = -0.023463847
    L_A_angular_acc = -0.0000671505
    R_K_sin	= 2897558.671
    R_K_sin2 = -32025.86344
    R_K_cos	= 2157252.931
    R_K_cos2 = -109165.2476
    R_A_sin	= -71998629.94
    R_A_sin2 = 2141922.119
    R_A_cos	= -1784745.391
    R_A_cos2 = 116129.3761
    L_K_sin	= -1195491.49
    L_K_sin2 = 6477.092878
    L_K_cos	= -1078375.077
    L_K_cos2 = 54000.18157
    L_A_sin	= -4449319.201
    L_A_sin2 = 120904.136
    L_A_cos	= -213089.455
    L_A_cos2 = 13968.37841
    R_K_angle_pow2 = 262.9921593
    R_K_angle_pow3 = 2.394708118
    R_A_angle_pow2 = -201.1641057
    R_A_angle_pow3 = -49.13991166
    L_K_angle_pow2 = -131.7789156
    L_K_angle_pow3 = -1.039683575
    L_A_angle_pow2 = -23.94185693
    L_A_angle_pow3 = -3.14094622
    R_K_angle_tan = -141.0814293
    R_A_angle_tan = 292339.5739
    L_K_angle_tan = -131.3505945
    L_A_angle_tan = 32351.77489
    L_A_moment = (Intercept + 
                R_K_prev_angle*rk_p_ang + R_K_angle*rk_ang + R_A_prev_angle*ra_p_ang + R_A_angle*ra_ang +
                L_K_prev_angle*lk_p_ang + L_K_angle*lk_ang + L_A_prev_angle*la_p_ang + L_A_angle*la_ang +
                R_K_angular_vel*(rk_ang-rk_p_ang)/(t2-t1) + R_K_angular_acc*((rk_ang-rk_p_ang)/(t2-t1) - (rk_p_ang-rk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                R_A_angular_vel*(ra_ang-ra_p_ang)/(t2-t1) + R_A_angular_acc*((ra_ang-ra_p_ang)/(t2-t1) - (ra_p_ang-ra_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_K_angular_vel*(lk_ang-lk_p_ang)/(t2-t1) + L_K_angular_acc*((lk_ang-lk_p_ang)/(t2-t1) - (lk_p_ang-lk_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) +
                L_A_angular_vel*(la_ang-la_p_ang)/(t2-t1) + L_A_angular_acc*((la_ang-la_p_ang)/(t2-t1) - (la_p_ang-la_p_p_ang)/(t1-t0))/((t2-t1)/2 + (t1-t0)/2) + 
                R_K_sin*sin(radians(rk_ang)) + R_K_sin2*sin(2*radians(rk_ang)) + R_K_cos*cos(radians(rk_ang)) + R_K_cos2*cos(2*radians(rk_ang)) + 
                R_A_sin*sin(radians(ra_ang)) + R_A_sin2*sin(2*radians(ra_ang)) + R_A_cos*cos(radians(ra_ang)) + R_A_cos2*cos(2*radians(ra_ang)) +
                L_K_sin*sin(radians(lk_ang)) + L_K_sin2*sin(2*radians(lk_ang)) + L_K_cos*cos(radians(lk_ang)) + L_K_cos2*cos(2*radians(lk_ang)) + 
                L_A_sin*sin(radians(la_ang)) + L_A_sin2*sin(2*radians(la_ang)) + L_A_cos*cos(radians(la_ang)) + L_A_cos2*cos(2*radians(la_ang)) +
                R_K_angle_pow2*rk_ang*rk_ang + R_K_angle_pow3*rk_ang*rk_ang*rk_ang + R_A_angle_pow2*ra_ang*ra_ang + R_A_angle_pow3*ra_ang*ra_ang*ra_ang +
                L_K_angle_pow2*lk_ang*lk_ang + L_K_angle_pow3*lk_ang*lk_ang*lk_ang + L_A_angle_pow2*la_ang*la_ang + L_A_angle_pow3*la_ang*la_ang*la_ang +
                R_K_angle_tan*tan(radians(rk_ang)) + R_A_angle_tan*tan(radians(ra_ang)) + L_K_angle_tan*tan(radians(lk_ang)) + L_A_angle_tan*tan(radians(la_ang))
                )
    return L_A_moment

Total_Used_Power = 0
Total_Regen_Power = 0
count_runs = 0
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
        count_runs = count_runs + 1
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
        drop_list = ["time","hip_flexion_r_moment","knee_angle_r_moment","ankle_angle_r_moment","hip_flexion_l_moment","knee_angle_l_moment","ankle_angle_l_moment"]
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
                    "hip_flexion_r_prev_prev","hip_flexion_r_prev","hip_flexion_r",
                    "knee_angle_r_prev_prev","knee_angle_r_prev", "knee_angle_r", 
                    "ankle_angle_r_prev_prev", "ankle_angle_r_prev","ankle_angle_r",
                    "hip_flexion_l_prev_prev","hip_flexion_l_prev","hip_flexion_l", 
                    "knee_angle_l_prev_prev","knee_angle_l_prev","knee_angle_l", 
                    "ankle_angle_l_prev_prev","ankle_angle_l_prev","ankle_angle_l"]
        df_angles = df_angles.drop(df_angles.columns.difference(drop_list), axis=1)
        #print(df_angles.describe)

        horizontal_stack = pd.concat([df_moment, df_angles], axis=1)
        cols = ['time_prev_prev','time_prev','time',
                'hip_flexion_r_moment','knee_angle_r_moment','ankle_angle_r_moment','hip_flexion_l_moment','knee_angle_l_moment','ankle_angle_l_moment',
                'hip_flexion_r_prev_prev','hip_flexion_r_prev','hip_flexion_r',
                'knee_angle_r_prev_prev','knee_angle_r_prev','knee_angle_r',
                'ankle_angle_r_prev_prev','ankle_angle_r_prev','ankle_angle_r',
                'hip_flexion_l_prev_prev','hip_flexion_l_prev','hip_flexion_l',
                'knee_angle_l_prev_prev','knee_angle_l_prev','knee_angle_l',
                'ankle_angle_l_prev_prev','ankle_angle_l_prev','ankle_angle_l']
        horizontal_stack = horizontal_stack[cols]
        col_add = ' P' + str(p_id) + '_Run-' + str(id_num)
        cols = ['time_prev_prev','time_prev','time',
                'hip_flexion_r_moment'+ col_add,'knee_angle_r_moment' + col_add,'ankle_angle_r_moment' + col_add,'hip_flexion_l_moment'+col_add,'knee_angle_l_moment' + col_add,'ankle_angle_l_moment' + col_add,
                'hip_flexion_r_prev_prev'+col_add,'hip_flexion_r_prev'+col_add,'hip_flexion_r'+col_add,
                'knee_angle_r_prev_prev' + col_add,'knee_angle_r_prev' + col_add,'knee_angle_r' + col_add,
                'ankle_angle_r_prev_prev' + col_add,'ankle_angle_r_prev' + col_add,'ankle_angle_r' + col_add,
                'hip_flexion_l_prev_prev'+col_add,'hip_flexion_l_prev'+col_add,'hip_flexion_l'+col_add,
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

        #Predicted Moments (Does Not Include Hip)
        horizontal_stack['knee_r_moment_predicted' + col_add] = moment_at_right_knee(horizontal_stack['time_prev_prev'],horizontal_stack['time_prev'],horizontal_stack['time'],
                                                                horizontal_stack['knee_angle_r_prev_prev' + col_add], horizontal_stack['knee_angle_r_prev' + col_add], horizontal_stack['knee_angle_r' + col_add],
                                                                horizontal_stack['ankle_angle_r_prev_prev' + col_add], horizontal_stack['ankle_angle_r_prev' + col_add], horizontal_stack['ankle_angle_r' + col_add],
                                                                horizontal_stack['knee_angle_l_prev_prev' + col_add], horizontal_stack['knee_angle_l_prev' + col_add], horizontal_stack['knee_angle_l' + col_add],
                                                                horizontal_stack['ankle_angle_l_prev_prev' + col_add], horizontal_stack['ankle_angle_l_prev' + col_add], horizontal_stack['ankle_angle_l' + col_add])
        horizontal_stack['ankle_r_moment_predicted' + col_add] = moment_at_right_ankle(horizontal_stack['time_prev_prev'],horizontal_stack['time_prev'],horizontal_stack['time'],
                                                                horizontal_stack['knee_angle_r_prev_prev' + col_add], horizontal_stack['knee_angle_r_prev' + col_add], horizontal_stack['knee_angle_r' + col_add],
                                                                horizontal_stack['ankle_angle_r_prev_prev' + col_add], horizontal_stack['ankle_angle_r_prev' + col_add], horizontal_stack['ankle_angle_r' + col_add],
                                                                horizontal_stack['knee_angle_l_prev_prev' + col_add], horizontal_stack['knee_angle_l_prev' + col_add], horizontal_stack['knee_angle_l' + col_add],
                                                                horizontal_stack['ankle_angle_l_prev_prev' + col_add], horizontal_stack['ankle_angle_l_prev' + col_add], horizontal_stack['ankle_angle_l' + col_add])
        
        horizontal_stack['knee_l_moment_predicted' + col_add] = moment_at_left_knee(horizontal_stack['time_prev_prev'],horizontal_stack['time_prev'],horizontal_stack['time'],
                                                                horizontal_stack['knee_angle_r_prev_prev' + col_add], horizontal_stack['knee_angle_r_prev' + col_add], horizontal_stack['knee_angle_r' + col_add],
                                                                horizontal_stack['ankle_angle_r_prev_prev' + col_add], horizontal_stack['ankle_angle_r_prev' + col_add], horizontal_stack['ankle_angle_r' + col_add],
                                                                horizontal_stack['knee_angle_l_prev_prev' + col_add], horizontal_stack['knee_angle_l_prev' + col_add], horizontal_stack['knee_angle_l' + col_add],
                                                                horizontal_stack['ankle_angle_l_prev_prev' + col_add], horizontal_stack['ankle_angle_l_prev' + col_add], horizontal_stack['ankle_angle_l' + col_add])
        horizontal_stack['ankle_l_moment_predicted' + col_add] = moment_at_left_ankle(horizontal_stack['time_prev_prev'],horizontal_stack['time_prev'],horizontal_stack['time'],
                                                                horizontal_stack['knee_angle_r_prev_prev' + col_add], horizontal_stack['knee_angle_r_prev' + col_add], horizontal_stack['knee_angle_r' + col_add],
                                                                horizontal_stack['ankle_angle_r_prev_prev' + col_add], horizontal_stack['ankle_angle_r_prev' + col_add], horizontal_stack['ankle_angle_r' + col_add],
                                                                horizontal_stack['knee_angle_l_prev_prev' + col_add], horizontal_stack['knee_angle_l_prev' + col_add], horizontal_stack['knee_angle_l' + col_add],
                                                                horizontal_stack['ankle_angle_l_prev_prev' + col_add], horizontal_stack['ankle_angle_l_prev' + col_add], horizontal_stack['ankle_angle_l' + col_add])
        
        horizontal_stack['knee_r_ang_vel'+col_add] = 0.0174533*(horizontal_stack['knee_angle_r'+col_add] - horizontal_stack['knee_angle_r_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_ang_vel'+col_add] = 0.0174533*(horizontal_stack['ankle_angle_r'+col_add] - horizontal_stack['ankle_angle_r_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_ang_vel'+col_add] = 0.0174533*(horizontal_stack['knee_angle_l'+col_add] - horizontal_stack['knee_angle_l_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_ang_vel'+col_add] = 0.0174533*(horizontal_stack['ankle_angle_l'+col_add] - horizontal_stack['ankle_angle_l_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])

        # horizontal_stack['hip_r_power'+col_add] = 0.0174533*horizontal_stack['hip_flexion_r_moment'+col_add]*(horizontal_stack['hip_flexion_r'+col_add] - horizontal_stack['hip_flexion_r_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_r_power'+col_add] = horizontal_stack['knee_angle_r_moment'+col_add]*horizontal_stack['knee_r_ang_vel'+col_add]
        horizontal_stack['knee_r_power_predicted'+col_add] = horizontal_stack['knee_r_moment_predicted'+col_add]*horizontal_stack['knee_r_ang_vel'+col_add]
        horizontal_stack['ankle_r_power'+col_add] = horizontal_stack['ankle_angle_r_moment'+col_add]*horizontal_stack['ankle_r_ang_vel'+col_add]
        horizontal_stack['ankle_r_power_predicted'+col_add] = horizontal_stack['ankle_r_moment_predicted'+col_add]*horizontal_stack['ankle_r_ang_vel'+col_add]
        # horizontal_stack['hip_l_power'+col_add] = 0.0174533*horizontal_stack['hip_flexion_l_moment'+col_add]*(horizontal_stack['hip_flexion_l'+col_add] - horizontal_stack['hip_flexion_l_prev'+col_add])/(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_power'+col_add] = horizontal_stack['knee_angle_l_moment'+col_add]*horizontal_stack['knee_l_ang_vel'+col_add]
        horizontal_stack['knee_l_power_predicted'+col_add] = horizontal_stack['knee_l_moment_predicted'+col_add]*horizontal_stack['knee_l_ang_vel'+col_add]
        horizontal_stack['ankle_l_power'+col_add] = horizontal_stack['ankle_angle_l_moment'+col_add]*horizontal_stack['ankle_l_ang_vel'+col_add]
        horizontal_stack['ankle_l_power_predicted'+col_add] = horizontal_stack['ankle_l_moment_predicted'+col_add]*horizontal_stack['ankle_l_ang_vel'+col_add]


        horizontal_stack['knee_r_power_human_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['knee_r_power_machine_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['ankle_r_power_human_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['ankle_r_power_machine_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['knee_l_power_human_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['knee_l_power_machine_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['ankle_l_power_human_required'+col_add] = pd.Series(dtype='float')
        horizontal_stack['ankle_l_power_machine_required'+col_add] = pd.Series(dtype='float')

        knee_r_max_moment = abs(horizontal_stack['knee_angle_r_moment'+col_add].min())
        if knee_r_max_moment < horizontal_stack['knee_angle_r_moment'+col_add].max():
            knee_r_max_moment = horizontal_stack['knee_angle_r_moment'+col_add].max()
        
        ankle_r_max_moment = abs(horizontal_stack['ankle_angle_r_moment'+col_add].min())
        if ankle_r_max_moment < horizontal_stack['ankle_angle_r_moment'+col_add].max():
            ankle_r_max_moment = horizontal_stack['ankle_angle_r_moment'+col_add].max()

        knee_l_max_moment = abs(horizontal_stack['knee_angle_l_moment'+col_add].min())
        if knee_l_max_moment < horizontal_stack['knee_angle_l_moment'+col_add].max():
            knee_l_max_moment = horizontal_stack['knee_angle_l_moment'+col_add].max()
        
        ankle_l_max_moment = abs(horizontal_stack['ankle_angle_l_moment'+col_add].min())
        if ankle_l_max_moment < horizontal_stack['ankle_angle_l_moment'+col_add].max():
            ankle_l_max_moment = horizontal_stack['ankle_angle_l_moment'+col_add].max()

        for q in range (horizontal_stack.shape[0]):
            if(abs(horizontal_stack['knee_angle_r_moment'+col_add][q])<(knee_r_max_moment*threshold)):
                horizontal_stack['knee_r_power_human_required'+col_add][q] = horizontal_stack['knee_r_power'+col_add][q]
                horizontal_stack['knee_r_power_machine_required'+col_add][q] = 0
            else:
                # print(horizontal_stack['knee_r_power'+col_add][q])
                # print(abs(horizontal_stack['knee_angle_r_moment'+col_add].max())*threshold)
                # print(horizontal_stack['knee_r_ang_vel'+col_add][q])
                horizontal_stack['knee_r_power_human_required'+col_add][q] = horizontal_stack['knee_r_power'+col_add][q]*(abs(knee_r_max_moment*threshold*horizontal_stack['knee_r_ang_vel'+col_add][q]/horizontal_stack['knee_r_power'+col_add][q]))
                horizontal_stack['knee_r_power_machine_required'+col_add][q] = horizontal_stack['knee_r_power'+col_add][q] - horizontal_stack['knee_r_power_human_required'+col_add][q]
            
            if(abs(horizontal_stack['ankle_angle_r_moment'+col_add][q])<(ankle_r_max_moment*threshold)):
                horizontal_stack['ankle_r_power_human_required'+col_add][q] = horizontal_stack['ankle_r_power'+col_add][q]
                horizontal_stack['ankle_r_power_machine_required'+col_add][q] = 0
            else:
                horizontal_stack['ankle_r_power_human_required'+col_add][q] = horizontal_stack['ankle_r_power'+col_add][q]*(abs(ankle_r_max_moment*threshold*horizontal_stack['ankle_r_ang_vel'+col_add][q]/horizontal_stack['ankle_r_power'+col_add][q]))
                horizontal_stack['ankle_r_power_machine_required'+col_add][q] = horizontal_stack['ankle_r_power'+col_add][q] - horizontal_stack['ankle_r_power_human_required'+col_add][q]
            
            if(abs(horizontal_stack['knee_angle_l_moment'+col_add][q])<(knee_l_max_moment*threshold)):
                horizontal_stack['knee_l_power_human_required'+col_add][q] = horizontal_stack['knee_l_power'+col_add][q]
                horizontal_stack['knee_l_power_machine_required'+col_add][q] = 0
            else:
                horizontal_stack['knee_l_power_human_required'+col_add][q] = horizontal_stack['knee_l_power'+col_add][q]*(abs(knee_l_max_moment*threshold*horizontal_stack['knee_l_ang_vel'+col_add][q]/horizontal_stack['knee_l_power'+col_add][q]))
                horizontal_stack['knee_l_power_machine_required'+col_add][q] = horizontal_stack['knee_l_power'+col_add][q] - horizontal_stack['knee_l_power_human_required'+col_add][q]
            
            if(abs(horizontal_stack['ankle_angle_l_moment'+col_add][q])<(ankle_l_max_moment*threshold)):
                horizontal_stack['ankle_l_power_human_required'+col_add][q] = horizontal_stack['ankle_l_power'+col_add][q]
                horizontal_stack['ankle_l_power_machine_required'+col_add][q] = 0
            else:
                horizontal_stack['ankle_l_power_human_required'+col_add][q] = horizontal_stack['ankle_l_power'+col_add][q]*(abs(ankle_l_max_moment*threshold*horizontal_stack['ankle_l_ang_vel'+col_add][q]/horizontal_stack['ankle_l_power'+col_add][q]))
                horizontal_stack['ankle_l_power_machine_required'+col_add][q] = horizontal_stack['ankle_l_power'+col_add][q] - horizontal_stack['ankle_l_power_human_required'+col_add][q]
        
        
        horizontal_stack['knee_r_power_regen'+col_add] = 0
        horizontal_stack['ankle_r_power_regen'+col_add] = 0
        horizontal_stack['knee_l_power_regen'+col_add] = 0
        horizontal_stack['ankle_l_power_regen'+col_add] = 0
        for r in range (horizontal_stack.shape[0]):
            if  horizontal_stack['knee_r_power_human_required'+col_add][r]<0:
                horizontal_stack['knee_r_power_regen'+col_add][r] = horizontal_stack['knee_r_power_human_required'+col_add][r]
                horizontal_stack['knee_r_power_human_required'+col_add][r] = 0
            if  horizontal_stack['knee_r_power_machine_required'+col_add][r]<0:
                horizontal_stack['knee_r_power_regen'+col_add][r] = horizontal_stack['knee_r_power_machine_required'+col_add][r]
                horizontal_stack['knee_r_power_machine_required'+col_add][r] = 0

            if  horizontal_stack['ankle_r_power_human_required'+col_add][r]<0:
                horizontal_stack['ankle_r_power_regen'+col_add][r] = horizontal_stack['ankle_r_power_human_required'+col_add][r]
                horizontal_stack['ankle_r_power_human_required'+col_add][r] = 0
            if  horizontal_stack['ankle_r_power_machine_required'+col_add][r]<0:
                horizontal_stack['ankle_r_power_regen'+col_add][r] = horizontal_stack['ankle_r_power_machine_required'+col_add][r]
                horizontal_stack['ankle_r_power_machine_required'+col_add][r] = 0

            if  horizontal_stack['knee_l_power_human_required'+col_add][r]<0:
                horizontal_stack['knee_l_power_regen'+col_add][r] = horizontal_stack['knee_l_power_human_required'+col_add][r]
                horizontal_stack['knee_l_power_human_required'+col_add][r] = 0
            if  horizontal_stack['knee_l_power_machine_required'+col_add][r]<0:
                horizontal_stack['knee_l_power_regen'+col_add][r] = horizontal_stack['knee_l_power_machine_required'+col_add][r]
                horizontal_stack['knee_l_power_machine_required'+col_add][r] = 0

            if  horizontal_stack['ankle_l_power_human_required'+col_add][r]<0:
                horizontal_stack['ankle_l_power_regen'+col_add][r] = horizontal_stack['ankle_l_power_human_required'+col_add][r]
                horizontal_stack['ankle_l_power_human_required'+col_add][r] = 0
            if  horizontal_stack['ankle_l_power_machine_required'+col_add][r]<0:
                horizontal_stack['ankle_l_power_regen'+col_add][r] = horizontal_stack['ankle_l_power_machine_required'+col_add][r]
                horizontal_stack['ankle_l_power_machine_required'+col_add][r] = 0

        horizontal_stack['knee_r_power_machine_required'+col_add] = horizontal_stack['knee_r_power_machine_required'+col_add]/(motr_output_eff*gear_train_eff)
        horizontal_stack['knee_l_power_machine_required'+col_add] = horizontal_stack['knee_l_power_machine_required'+col_add]/(motr_output_eff*gear_train_eff)
        horizontal_stack['ankle_r_power_machine_required'+col_add] = horizontal_stack['ankle_r_power_machine_required'+col_add]/(motr_output_eff*gear_train_eff)
        horizontal_stack['ankle_l_power_machine_required'+col_add] = horizontal_stack['ankle_l_power_machine_required'+col_add]/(motr_output_eff*gear_train_eff)
        horizontal_stack['knee_r_power_regen'+col_add] = horizontal_stack['knee_r_power_regen'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['knee_l_power_regen'+col_add] = horizontal_stack['knee_l_power_regen'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['ankle_r_power_regen'+col_add] = horizontal_stack['ankle_r_power_regen'+col_add]*motor_regen_eff*gear_train_eff
        horizontal_stack['ankle_l_power_regen'+col_add] = horizontal_stack['ankle_l_power_regen'+col_add]*motor_regen_eff*gear_train_eff


        # horizontal_stack['hip_r_energy'+col_add] = horizontal_stack['hip_r_power'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_r_energy'+col_add] = horizontal_stack['knee_r_power'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_r_energy_predicted'+col_add] = horizontal_stack['knee_r_power_predicted'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_energy'+col_add] = horizontal_stack['ankle_r_power'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_energy_predicted'+col_add] = horizontal_stack['ankle_r_power_predicted'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        # horizontal_stack['hip_l_energy'+col_add] = horizontal_stack['hip_l_power'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_energy'+col_add] = horizontal_stack['knee_l_power'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_energy_predicted'+col_add] = horizontal_stack['knee_l_power_predicted'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_energy'+col_add] = horizontal_stack['ankle_l_power'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_energy_predicted'+col_add] = horizontal_stack['ankle_l_power_predicted'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        
        horizontal_stack['knee_r_energy_human_required'+col_add] = horizontal_stack['knee_r_power_human_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_r_energy_machine_required'+col_add] = horizontal_stack['knee_r_power_machine_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_r_energy_regen'+col_add] = -horizontal_stack['knee_r_power_regen'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_energy_human_required'+col_add] = horizontal_stack['ankle_r_power_human_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_energy_machine_required'+col_add] = horizontal_stack['ankle_r_power_machine_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_r_energy_regen'+col_add] = -horizontal_stack['ankle_r_power_regen'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_energy_human_required'+col_add] = horizontal_stack['knee_l_power_human_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_energy_machine_required'+col_add] = horizontal_stack['knee_l_power_machine_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['knee_l_energy_regen'+col_add] = -horizontal_stack['knee_l_power_regen'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_energy_human_required'+col_add] = horizontal_stack['ankle_l_power_human_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_energy_machine_required'+col_add] = horizontal_stack['ankle_l_power_machine_required'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])
        horizontal_stack['ankle_l_energy_regen'+col_add] = -horizontal_stack['ankle_l_power_regen'+col_add]*(horizontal_stack['time']-horizontal_stack['time_prev'])

        # horizontal_stack['Energy_At_Instance_'+col_add] = (absolute(horizontal_stack['hip_r_energy'+col_add]) + absolute(horizontal_stack['knee_r_energy'+col_add]) + 
        #                                             absolute(horizontal_stack['ankle_r_energy'+col_add]) + absolute(horizontal_stack['hip_l_energy'+col_add]) +
        #                                             absolute(horizontal_stack['knee_l_energy'+col_add]) + absolute(horizontal_stack['ankle_l_energy'+col_add]))
        horizontal_stack['Energy_At_Instance_Human_'+col_add] = (absolute(horizontal_stack['knee_r_energy_human_required'+col_add]) + absolute(horizontal_stack['ankle_r_energy_human_required'+col_add]) + 
                                                    absolute(horizontal_stack['knee_l_energy_human_required'+col_add]) + absolute(horizontal_stack['ankle_l_energy_human_required'+col_add]))
        horizontal_stack['Energy_At_Instance_Machine_'+col_add] = (absolute(horizontal_stack['knee_r_energy_machine_required'+col_add]) + absolute(horizontal_stack['ankle_r_energy_machine_required'+col_add]) + 
                                                    absolute(horizontal_stack['knee_l_energy_machine_required'+col_add]) + absolute(horizontal_stack['ankle_l_energy_machine_required'+col_add]))    
        horizontal_stack['Energy_At_Instance_Regen_'+col_add] = (absolute(horizontal_stack['knee_r_energy_regen'+col_add]) + absolute(horizontal_stack['ankle_r_energy_regen'+col_add]) + 
                                                    absolute(horizontal_stack['knee_l_energy_regen'+col_add]) + absolute(horizontal_stack['ankle_l_energy_regen'+col_add]))                                  
        
        cols = ['time_prev_prev','time_prev','time',
                'knee_r_power_human_required'+col_add, 'knee_r_power_machine_required'+col_add, 'knee_r_power_regen'+col_add,
                'ankle_r_power_human_required'+col_add, 'ankle_r_power_machine_required'+col_add, 'ankle_r_power_regen'+col_add,
                'knee_l_power_human_required'+col_add, 'knee_l_power_machine_required'+col_add, 'knee_l_power_regen'+col_add,
                'ankle_l_power_human_required'+col_add, 'ankle_l_power_machine_required'+col_add, 'ankle_l_power_regen'+col_add,
                'Energy_At_Instance_Human_'+col_add,'Energy_At_Instance_Machine_'+col_add, 'Energy_At_Instance_Regen_'+col_add]
        horizontal_stack = horizontal_stack.drop(horizontal_stack.columns.difference(cols), axis=1)
        horizontal_stack['Energy_Over_Time_Human_'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Over_Time_Machine_'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_Over_Time_Regen_'+col_add] = pd.Series(dtype='float')

        horizontal_stack['Duration_in_seconds_'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Duration_in_seconds_'+col_add][0] = horizontal_stack['time'].max()
        horizontal_stack['Energy_sum'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_sum'+col_add][0] = horizontal_stack['Energy_At_Instance_Human_'+col_add].sum()
        horizontal_stack['Energy_average'+col_add] = pd.Series(dtype='float')
        horizontal_stack['Energy_average'+col_add][0] = horizontal_stack['Energy_sum'+col_add][0]/horizontal_stack['Duration_in_seconds_'+col_add][0]

        for p in range (horizontal_stack.shape[0]):
            if p==0:
                horizontal_stack['Energy_Over_Time_Human_'+col_add][p] = horizontal_stack['Energy_At_Instance_Human_'+col_add][0]
                horizontal_stack['Energy_Over_Time_Machine_'+col_add][p] = horizontal_stack['Energy_At_Instance_Machine_'+col_add][0]
                horizontal_stack['Energy_Over_Time_Regen_'+col_add][p] = horizontal_stack['Energy_At_Instance_Regen_'+col_add][0]
            else:
                horizontal_stack['Energy_Over_Time_Human_'+col_add][p] = horizontal_stack['Energy_Over_Time_Human_'+col_add][p-1] + horizontal_stack['Energy_At_Instance_Human_'+col_add][p] 
                horizontal_stack['Energy_Over_Time_Machine_'+col_add][p] = horizontal_stack['Energy_Over_Time_Machine_'+col_add][p-1] + horizontal_stack['Energy_At_Instance_Machine_'+col_add][p] 
                horizontal_stack['Energy_Over_Time_Regen_'+col_add][p] = horizontal_stack['Energy_Over_Time_Regen_'+col_add][p-1] + horizontal_stack['Energy_At_Instance_Regen_'+col_add][p] 
        Total_Used_Power = Total_Used_Power + horizontal_stack['Energy_Over_Time_Machine_'+col_add][horizontal_stack.shape[0]-1]/horizontal_stack['time'][horizontal_stack.shape[0]-1]
        Total_Regen_Power = Total_Regen_Power +  horizontal_stack['Energy_Over_Time_Regen_'+col_add][horizontal_stack.shape[0]-1]/horizontal_stack['time'][horizontal_stack.shape[0]-1]
        # print(id_num)
        # print(first_timestamp)
        # print(horizontal_stack.describe)
        # horizontal_stack = horizontal_stack[cols]
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
horizontal_stack_overall.to_csv("Energy_and_Power_with_regen_based_on_threshold_80%.csv", sep=',')
Total_Used_Power = Total_Used_Power/count_runs
Total_Regen_Power = Total_Regen_Power/count_runs
print("Total Used Power On Average: ", Total_Used_Power)
print("Total Regenerated Power On Average: ", Total_Regen_Power)
print("Total Count: ", count_runs)