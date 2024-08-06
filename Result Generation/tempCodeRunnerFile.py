Total_Used_Power = Total_Used_Power + horizontal_stack['Energy_Over_Time_Machine_'+col_add][horizontal_stack.shape[0]-1]/horizontal_stack['time'][horizontal_stack.shape[0]-1]
        Total_Regen_Power = Total_Regen_Power +  horizontal_stack['Energy_Over_Time_Regen_'+col_add][horizontal_stack.shape[0]-1]/horizontal_stack['time'][horizontal_stack.shape[0]-1]
        