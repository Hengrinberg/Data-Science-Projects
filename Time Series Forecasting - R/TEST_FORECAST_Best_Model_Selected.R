
# ---------------------------   Fox Home Prediction - subgroup level (best model selected)   ---------------------------------------------

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
#DF_subgroups <- read_excel("C:/Users/heng/Desktop/input_data_fox_2.xlsx")
#DF_subgroups <- read_excel("C:/Users/heng/Desktop/valid_input_data_for_forecasting_R.xlsx")
#DF_subgroups <- read_excel("C:/Users/heng/Desktop/input_data_for_forecasting_including_foxhome.xlsx")
DF_subgroups <- read_excel("C:/Users/heng/Desktop/fox_data/fox_home_input_data_subgroup_level.xlsx")


DF_subgroups_split_by_subgroup = split( DF_subgroups , f = DF_subgroups$SubGrp_ID)


# prepare the data to the form that required by forecasting algorithms
list_of_ready_dataframes_subgroups = list()

for(i in DF_subgroups_split_by_subgroup){

  i = as.data.frame(i[,c(1,2,3,4,5)])
  as.Date(i$Month)
  full_dataframe_subgroup = data.frame(SubGrp_ID  = c(rep(0,32)),
                                       Month = c('2015-01-01','2015-02-01','2015-03-01','2015-04-01','2015-05-01',
                                                 '2015-06-01','2015-07-01','2015-08-01','2015-09-01','2015-10-01',
                                                 '2015-11-01','2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01',
                                                 '2016-06-01','2016-07-01','2016-08-01','2016-09-01','2016-10-01',
                                                 '2016-11-01','2016-12-01','2017-01-01','2017-02-01','2017-03-01',
                                                 '2017-04-01','2017-05-01','2017-06-01','2017-07-01','2017-08-01'),
                                       Qty  = c(rep(0,32)),
                                       year_ = c(rep('2015',12),rep('2016',12),rep('2017',8)),
                                       month_ = c(1,2,3,4,5,6,7,8,9,10,11,12,
                                                  1,2,3,4,5,6,7,8,9,10,11,12,
                                                  1,2,3,4,5,6,7,8))
  
  as.Date(full_dataframe_subgroup$Month)
  if (nrow(i) == 32) {    colnames(i) <- c("Degem","Month","Qty","year_", "month_")
  list_of_ready_dataframes_subgroups = append(list_of_ready_dataframes_subgroups,list(i))}
  else{if(nrow(i) > 32) {i = i[1:32,]}
    subgroup_vec = c(rep(i$SubGrp_ID[1],32)) 
    mergerd_df = merge(x = i, y = full_dataframe_subgroup, by = c('year_','month_'), all.y = TRUE)
    mergerd_df$Qty.x[is.na(mergerd_df$Qty.x)] =  0
    mergerd_df$SubGrp_ID = subgroup_vec
    print(mergerd_df)
    ready_df = mergerd_df[,c(9,7,5,1,2)]
    colnames(ready_df) <- c("SubGrp_ID","month","Qty","year_","month_")
    list_of_ready_dataframes_subgroups = append(list_of_ready_dataframes_subgroups , list(ready_df))
  }
}


#------------------------------ exponential smoothing - forecasting --------------------------------------


# convert to a time series
list_of_error_result = list()
seasonality_classification_result = list()
list_of_forecasting_models = list("AZZ","AAA","AAN","ANN") # list of different exponential smoothing forecast models
best_actual_predicted_data_list = list()


# function that calculates MAPE error
mape <- function(actual,pred){
  mape <- mean(abs((actual - pred)/actual))*100
  return (mape)
}



for (df_ in  list_of_ready_dataframes_subgroups){
  list_of_mape_errors = list()
  actual_predicted_data_list = list()
  for (item_ in list_of_forecasting_models){
  df_[order(df_[,4],df_[,5],decreasing=FALSE),]
  real_values = c(df_[9:32,3])
  
  vec_sep15_aug_16 = df_[9:20,c(2,3)]
  vec_sep16_aug_17 = df_[21:32,c(2,3)]
  train_ = df_[9:32,]
  
  # seasonality classification
  vec_sep15_aug_16 = as.vector(vec_sep15_aug_16[,2]) # take just the sales values as a vector
  vec_sep16_aug_17 = as.vector(vec_sep16_aug_17[,2]) # take just the sales values as a vector
  correlation_ = cor(vec_sep15_aug_16, vec_sep16_aug_17)
  correlation_ = ifelse(is.na(correlation_), 0, correlation_)
  
  # variance test - check if the values inside  vec_sep16_aug_17 are different from each other in order to determine seasonality
  average_vec_sep16_aug_17 = mean(vec_sep16_aug_17) # the average of the series
  average_vec_sep16_aug_17 = ifelse(is.na(average_vec_sep16_aug_17), 10000000 , average_vec_sep16_aug_17)
  average_vec_sep16_aug_17 = ifelse(average_vec_sep16_aug_17 <= 0, 10000000 , average_vec_sep16_aug_17)
  std_vec_sep16_aug_17 = sd(vec_sep16_aug_17)
  average_vec_sep16_aug_17 = ifelse(std_vec_sep16_aug_17 < 0, 0 , average_vec_sep16_aug_17)
  variance_value = (std_vec_sep16_aug_17 / average_vec_sep16_aug_17)
  
  result = 0
  if (correlation_ >= 0.6 && variance_value >= 0.3){
    result = as.character("seasonal")
  }
  else{ result = as.character("not seasonal")}
  
  classification = data.frame(rep(result,nrow(df_)))
  df_ = cbind(df_,classification) # add classification column to each degem dataframe
  
  seasonality_classification_result = append(seasonality_classification_result,as.character(result))
  
  month = c(df_[25:32,2])
  #ts_object = ts(train_$Qty, start = 2015 , frequency = 12)
  ts_object = ts(train_$Qty, start = c(2015,9), end=c(2017, 8) , frequency = 12)
  print(ts_object)
  class(ts_object)
  
  # plot the ts_object
  plot.ts(ts_object, xlab = 'Time', ylab = 'num_units_sold')
  

  item_ = item_[[1]]
  Num_Units_SoldF <- forecast:::ets(ts_object, model = item_)
  Num_Units_SoldF$residuals
  avg_residual_error = mean(Num_Units_SoldF$residuals)
  fitted = fitted.values(Num_Units_SoldF)
  Predicted_Num_Units_Sold = forecast:::forecast(Num_Units_SoldF)
  #plot(forecast(fit,h=24))
  plot(Predicted_Num_Units_Sold)
  
  
  #Make forecasts
  #Predicted_Num_Units_Sold <- forecast:::forecast.HoltWinters(Num_Units_SoldF, h=24) # predict 24 months ahead
  #Predicted_Num_Units_Sold <- forecast:::forecast.ets(Num_Units_SoldF,model = "AAM"???) # predict 8 months ahead
  plot(Predicted_Num_Units_Sold) # plot the forecast
  df_predicted = as.data.frame(Predicted_Num_Units_Sold)
  df_predicted$`Point Forecast`[df_predicted$`Point Forecast` < 0] = 0 # set all negative predicted values to be 0
  

  
  mape_error_value = mape(c(train_$Qty),c(fitted))
  list_of_mape_errors = append(list_of_mape_errors, mape_error_value)
  
  actual_predicted_data = data.frame( Actual = c(rep(0,48)),
                                      Predicted = c(rep(0,48)))

  additional_real_values_vec = c(rep(0,24))
  additional_predicted_values_vec = c(rep(0,24))
  ready_actual_vec = as.data.frame(c(real_values,additional_real_values_vec))
  
  
  predicted_vec_total = c(c(fitted),c(df_predicted[,1]))
  actual_predicted_data["Actual"] = as.data.frame(ready_actual_vec)
  actual_predicted_data["Predicted"] = as.data.frame(predicted_vec_total)
  actual_predicted_data_list = append(actual_predicted_data_list, list(actual_predicted_data))
  }
  # find the forecasting model with the minimum mape error value
  min_index = which.min(list_of_mape_errors) # find the index of the minimum mape value error
  best_actual_predicted_data_list = append(best_actual_predicted_data_list,
                                           list(actual_predicted_data_list[[min_index[[1]]]]))
}

# give a name to each item inside list_of_error_result 
names_2 = names(DF_subgroups_split_by_subgroup)
names(best_actual_predicted_data_list) = names_2


# ready_output is a dataframe that contains all the actual and predicted data for each degem and enables to explore the results. 
output = list()
for (i in (1:length(best_actual_predicted_data_list))){
  subgroup_name = names_2[[i]]
  name_df = data.frame(subgroup_name = rep(subgroup_name,nrow(best_actual_predicted_data_list[[i]])))
  new_df = cbind(name_df,best_actual_predicted_data_list[[i]])
  output = append(output,list(new_df))
  ready_output =  Reduce(rbind, output) # concatenate all dataframes into one dataframe
  
}




# write the output dataframe to excel
library(xlsx)
write.xlsx(ready_output, "C:/Users/heng/Desktop/first_forecast_results_Fox_Home_subgroup_level_best_model_selected2.xlsx")















