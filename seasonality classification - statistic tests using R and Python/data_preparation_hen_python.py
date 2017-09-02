import pandas as pd
import numpy as np
from pandas import ExcelWriter




# Read the excel sheet to pandas dataframe
df = pd.read_excel("C:\Users\heng\Desktop\data_for_project.xlsx", converters={'Year-Month':str,'Year-Week':str}) # converters specify the type of each column
print type(df)

# show the column names
col_names = list(df.columns.values) 
print col_names
print df.dtypes

#replace all null values in 0
df.fillna({'POS Units (Net)': 0}, inplace=True)
df.fillna({'Shipment Units': 0}, inplace=True)

#replace all negative values in 0
df[df['POS Units (Net)'] < 0] = 0
df[df['Shipment Units'] < 0] = 0

#validate all the replacements
print " num of null's:" + " " + str(df.isnull().sum())


# basic statistics about the dataframe
print df.describe()


#group the data by sku
df = df.groupby(['SKU'])

# empty list that will contain all the time serieses
serieses =[]

# list of unique SKU's 
keys = df.groups.keys()

# store in serieses all the time serieses that have more than one observation
for key  in keys:
	if len(df.get_group(key).index) > 1:
		serieses.append(df.get_group(key))
		
				
# A function which calculates the difference between two dates in weeks	
def week_diff(df_,ind,counter):
	current_week  = int(df_.iloc[ind,1][-2:])  # store the last 2 digits from  'year - week' column for the current observation
	current_year = int(df_.iloc[ind,1][:4])    # store the first 4 digits from  'year - week' column for the current observation
	previous_sale_week = int(df_.iloc[ind - counter,1][-2:]) # store the last 2 digits from  'year - week' column for the last pos > 0
	previous_sale_year = int(df_.iloc[ind - counter,1][:4])  # store the first 4 digits from  'year - week' column for the last pos > 0
	diff = 0	
	if current_year == previous_sale_year:
		diff = current_week -  previous_sale_week 
	else:
		year_diff = current_year - previous_sale_year
		if current_week -  previous_sale_week < 0:
			year_diff  -= 1
		diff = (current_week ) + (year_diff * 52) + (52 - previous_sale_week) 
		
	return diff
	
	

			
	
	
#  A function which add to each dataframe in serieses list the columns: num_of_obs , pos_lags_in_weeks and shipment_lags_in_weeks 	
def add_three_columns():			
	for df_ in serieses:
		num_of_obs = len(df_.index)
		num_of_obs_column = [num_of_obs] * num_of_obs 
		pos_lags_in_weeks = [0]
		shipment_lags_in_weeks = [0]
		month_ = []
		year_ = []
		for ind in range (1 ,len(df_.index)):
			counter = 0
			last_value = 0
			if df_.iloc[ind - 1,10] > 0:
				pos_lags_in_weeks.append(week_diff(df_,ind,counter + 1))#df_[ind,1][-2:] - df_[ind - 1,1][-2:])
			else:
				while (ind - counter - 1) >= 0  and df_.iloc[ind - counter - 1,10] == 0  :
					last_value = df_.iloc[ind - counter -1 ,10]
					counter += 1
					
				if ind - counter > 0:
					pos_lags_in_weeks.append(week_diff(df_,ind,counter)) #df_[ind,1][-2:] - df_[ind - counter,1][-2:])
				else:
					pos_lags_in_weeks.append(0)
						
		for ind in range (1 ,len(df_.index)):
			counter = 0
			last_value = 0
			if df_.iloc[ind - 1,11] > 0:
				shipment_lags_in_weeks.append(1)
			else:
				while (ind - counter - 1) >= 0 and df_.iloc[ind - counter - 1,11] == 0 :
					last_value = df_.iloc[ind - counter -1 ,11]
					counter += 1

				if ind - counter > 0:
					shipment_lags_in_weeks.append(counter)
				else:
					shipment_lags_in_weeks.append(0)
				
		df_['num_of_obs'] = num_of_obs_column
		df_['pos_lags_in_weeks'] = pos_lags_in_weeks
		df_['shipment_lags_in_weeks'] = shipment_lags_in_weeks
		
		
	return serieses

result =  add_three_columns() # a list that contains all the dataframes after the transformations



# export all the dataframes to a xlsx file
def save_xls(list_dfs, xls_path):
    writer = ExcelWriter(xls_path)
    for n, df in enumerate(list_dfs):
        df.to_excel(writer,'sheet%s' % n)
    writer.save()

save_xls(result, "C:\Users\heng\Desktop\clean_data5.xlsx")






