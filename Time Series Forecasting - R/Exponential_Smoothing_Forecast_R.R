
# ------------------------------------ Data Preparation - SubGroup level ---------------------------------------------

library(readxl)
install.packages("xlsx")
install.packages("rJava")
library(xlsx)
library(rJava)
install.packages("openxlsx")
library(openxlsx)
#library(data.table)
install.packages("data.table")
library("data.table")
install.packages("dplyr")
library(dplyr)
install.packages("stringi")
library(stringi)
install.packages("forecast")
library(forecast)

# read the excel file
DF_subgroups <- read_excel("C:/Users/heng/Desktop/input_data_for_forecasting_including_foxhome.xlsx")

DF_subgroups_split_by_degem = split( DF_subgroups , f = DF_subgroups$SubGrp_ID)


# prepare the data to the form that required by forecasting algorithms
list_of_ready_dataframes_subgroups = list()
for(i in DF_subgroups_split_by_degem){
  i = as.data.frame(i[,c(1,2,3,4,5)])
  #year_ = substr(i$Month,1,4)
  #month_ = substr(i$Month,6,2)
  as.Date(i$Month)
  full_dataframe_subgroup = data.frame(SubGrp_ID  = c(rep(0,32)),
                                       Month = c('2015-01-01','2015-02-01','2015-03-01','2015-04-01','2015-05-01',
                                       '2015-06-01','2015-07-01','2015-08-01','2015-09-01','2015-10-01',
                                       '2015-11-01','2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01',
                                       '2016-06-01','2016-07-01','2016-08-01','2016-09-01','2016-10-01',
                                       '2016-11-01','2016-12-01','2017-01-01','2017-02-01','2017-03-01',
                                       '2017-04-01','2017-05-01','2017-06-01','2017-07-01','2017-08-01'),
                                       Qty  = c(rep(0,32)),
                                       year = c(rep('2015',12),rep('2016',12),rep('2017',8)),
                                       month_ = c(1,2,3,4,5,6,7,8,9,10,11,12,
                                                 1,2,3,4,5,6,7,8,9,10,11,12,
                                                 1,2,3,4,5,6,7,8))
  
  as.Date(full_dataframe_subgroup$Month)
  if (nrow(i) == 32) {    colnames(i) <- c("SubGrp_ID","Month","Qty","year", "month_")
  list_of_ready_dataframes_subgroups = append(list_of_ready_dataframes_subgroups,list(i))}
  else{if(nrow(i) > 32) {i = i[1:32,]}
    sub_group_vec = c(rep(i$SubGrp_ID[1],32)) 
    mergerd_df = merge(x = i, y = full_dataframe_subgroup, by = c('year','month_'), all.y = TRUE)
    mergerd_df$Qty.x[is.na(mergerd_df$Qty.x)] =  0
    mergerd_df$SubGrp_ID = sub_group_vec
    print(mergerd_df)
    ready_df = mergerd_df[,c(9,7,5,1,2)]
    colnames(ready_df) <- c("SubGrp_ID","month","Qty","year","month_")
    list_of_ready_dataframes_subgroups = append(list_of_ready_dataframes_subgroups , list(ready_df))
  }
}


# give a name to each dataframe inside list_of_ready_dataframes_subgroups list 
names = names(DF_subgroups_split_by_degem)
names(list_of_ready_dataframes_subgroups) = names



#------------------------------ exponential smoothing - forecasting --------------------------------------


# convert to a time series
list_of_error_result = list()
actual_predicted_data_list = list()

for (df_ in  list_of_ready_dataframes_subgroups){
  train_ = df_[df_$year == 2015 | df_$year == 2016,]
  real_values = c(df_[df_$year == 2017,]$Qty)

  month = c(df_[25:32,2])
  
  ts_object = ts(train_$Qty, start = 2015 , frequency = 12)
  print(ts_object)
  class(ts_object)
  
  # plot the ts_object
  plot.ts(ts_object, xlab = 'Time', ylab = 'num_units_sold')
  Num_Units_SoldF <- HoltWinters(ts_object)
  
  
  #Make forecasts
  Predicted_Num_Units_Sold <- forecast:::forecast.HoltWinters(Num_Units_SoldF, h=24) # predict 24 months ahead
  plot(Predicted_Num_Units_Sold) # plot the forecast
  df_predicted = as.data.frame(Predicted_Num_Units_Sold)
  df_predicted$`Point Forecast`[df_predicted$`Point Forecast` < 0] = 0 # set all negative predicted values to be 0
  
  
  
  # calculate the MSE( mean square error between the actual value and the prediction)
  real_values = unlist(real_values, recursive = TRUE, use.names = FALSE) # convert real_values from list to vector
  Mean_Square_Error =  sum((real_values - c(df_predicted[,1]))^2 )/8
  #Mape = sum(abs((real_values - c(df_predicted[,1])/real_values))) * 100 / 8
  list_of_error_result = append(list_of_error_result , list(Mean_Square_Error))
  actual_predicted_data = data.frame(month = '2017-01-01','2017-02-01','2017-03-01',
                                     '2017-04-01','2017-05-01','2017-06-01','2017-07-01','2017-08-01',
                                     Actual = c(rep(0,8)),
                                     Predicted = c(rep(0,8)),
                                     Error = c(rep(0,8)))
  
  actual_predicted_data["Actual"] = as.data.frame(real_values)
  actual_predicted_data["Predicted"] = as.data.frame(df_predicted[,1])
  actual_predicted_data["Error"] = c(actual_predicted_data$Actual) - c(actual_predicted_data$Predicted)
  actual_predicted_data_list = append(actual_predicted_data_list, list(actual_predicted_data))
}


# give a name to each item inside list_of_error_result 
names_2 = names(DF_subgroups_split_by_degem)
names(list_of_error_result) = names_2
names(actual_predicted_data_list) = names_2


# ready_output is a dataframe that contains all the actual and predicted data for each sub_group and enables to explore the results. 
output = list()
for (i in (1:length(actual_predicted_data_list))){
  subgroup_name = names_2[[i]]
  name_df = data.frame(subgroup_name = rep(subgroup_name,nrow(actual_predicted_data_list[[i]])))
  new_df = cbind(name_df,actual_predicted_data_list[[i]])
  output = append(output,list(new_df))
  ready_output =  Reduce(rbind, output) # concatenate all dataframes into one dataframe
}


# dataframe that contains the subgroup names and the mean square error of each sub group - export to excel 
test_result = data.frame(subgroup_name = c(rep("a",length(names_2))) 
                         , Error = c(rep(0,length(list_of_error_result))))

test_result$subgroup_name = names_2
test_result$Error = list_of_error_result


# write the test_result dataframe to excel
library(xlsx)
write.xlsx(test_result, "C:/Users/heng/Desktop/first_forecast_results_with_foxhome.xlsx")


# write the output dataframe to excel
library(xlsx)
write.xlsx(ready_output, "C:/Users/heng/Desktop/first_forecast_results_all_data_with_fox_home.xlsx")





test_v = DF_subgroups_split_by_degem[15]







