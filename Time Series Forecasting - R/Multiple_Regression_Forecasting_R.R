
# -----------   Fox Prediction - subgroup level all hierarchy (best model selected) - MLR algorithm   ---------------------------------------------

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
DF_subgroups <- read_excel("C:/Users/heng/Desktop/fox_data/Fox_input_data_all_hierarchy.xlsx")


DF_subgroups_split_by_subgroup = split(DF_subgroups,list(DF_subgroups$MainDepartmentID,DF_subgroups$MainGroupID,
                                                         DF_subgroups$GroupID,DF_subgroups$SubGrp_ID))

# remove all empty dataframes 
DF_subgroups_split_by_subgroup = DF_subgroups_split_by_subgroup[sapply(DF_subgroups_split_by_subgroup, function(x) dim(x)[1]) > 0]

# prepare the data to the form that required by forecasting algorithms
list_of_ready_dataframes_subgroups = list()

for(i in DF_subgroups_split_by_subgroup){
  i = as.data.frame(i[,c(1,2,3,4,5,6,7,8)])
  as.Date(i$Month)
  full_dataframe_subgroup = data.frame(SubGrp_ID  = c(rep(0,32)),
                                       Month = c('2015-01-01','2015-02-01','2015-03-01','2015-04-01','2015-05-01',
                                                 '2015-06-01','2015-07-01','2015-08-01','2015-09-01','2015-10-01',
                                                 '2015-11-01','2015-12-01','2016-01-01','2016-02-01','2016-03-01','2016-04-01','2016-05-01',
                                                 '2016-06-01','2016-07-01','2016-08-01','2016-09-01','2016-10-01',
                                                 '2016-11-01','2016-12-01','2017-01-01','2017-02-01','2017-03-01',
                                                 '2017-04-01','2017-05-01','2017-06-01','2017-07-01','2017-08-01'),
                                       MainDepartmentID = c(rep(0,32)),
                                       MainGroupID = c(rep(0,32)),
                                       GroupID = c(rep(0,32)),
                                       
                                       Qty  = c(rep(0,32)),
                                       year = c(rep('2015',12),rep('2016',12),rep('2017',8)),
                                       month_ = c(1,2,3,4,5,6,7,8,9,10,11,12,
                                                  1,2,3,4,5,6,7,8,9,10,11,12,
                                                  1,2,3,4,5,6,7,8))
  
  as.Date(full_dataframe_subgroup$Month)
  if (nrow(i) == 32) {    colnames(i) <- c("MainDepartmentID", "MainGroupID","GroupID", "SubGrp_ID","Month","year","month_" ,"QTY")
  list_of_ready_dataframes_subgroups = append(list_of_ready_dataframes_subgroups,list(i))}
  else{if(nrow(i) > 32) {i = i[1:32,]}
    subgroup_vec = c(rep(i$SubGrp_ID[1],32))
    MainDepartmentID_vec = c(rep(i$MainDepartmentID[1],32))
    MainGroupID_vec = c(rep(i$MainGroupID[1],32))
    GroupID_vec = c(rep(i$GroupID[1],32))
    mergerd_df = merge(x = i, y = full_dataframe_subgroup, by = c('year','month_'), all.y = TRUE)
    mergerd_df$Qty.x[is.na(mergerd_df$Qty.x)] =  0
    mergerd_df$SubGrp_ID = subgroup_vec
    mergerd_df$MainDepartmentID = MainDepartmentID_vec
    mergerd_df$MainGroupID = MainGroupID_vec
    mergerd_df$GroupID.x = GroupID_vec
    print(mergerd_df)
    ready_df = mergerd_df[,c(16,17,5,15,10,1,2,8)]
    colnames(ready_df) <- c("MainDepartmentID", "MainGroupID","GroupID", "SubGrp_ID","Month","year","month_" ,"QTY")
    list_of_ready_dataframes_subgroups = append(list_of_ready_dataframes_subgroups , list(ready_df))
  }
}


#------------------------------ MLR ALGORITHMs - forecasting --------------------------------------


best_actual_predicted_data_list = list()


for (df_ in  list_of_ready_dataframes_subgroups){
  actual_predicted_data_list = list()
  df_[order(df_[,6],df_[,7],decreasing=FALSE),]

  y1 = df_[9:20,c(2,3)]
  y2 = df_[21:32,c(2,3)]
  real_values = c(df_[9:32,8])
  Ratio_2015_2016 = sum(y2) / max(1,sum(y1))
  
  
  # check if all real vaues are less than 5 the prediction vector will be vector of ones
  if(sum(y2) < 100){
    actual_predicted_data = data.frame( Actual = c(rep(0,48)),
                                        Predicted = c(rep(0,48)))
    
    additional_real_values_vec = c(rep(0,24))
    ready_actual_vec = as.data.frame(c(real_values,additional_real_values_vec))
    
    actual_predicted_data["Actual"] = as.data.frame(ready_actual_vec)
    best_actual_predicted_data_list = append(best_actual_predicted_data_list, list(actual_predicted_data))
  } 
  
  
  else{

      vec_sep15_aug_16 = df_[9:20,c(5,8)]
      vec_sep16_aug_17 = df_[21:32,c(5,8)]
      train_ = df_[9:32,c(5,8)]
      
      train_trend_vec = data.frame( trend = 1:24)
      train_season_vec = 0
      tmp_season = list()
      for (i in 1:12){
        value_ = mean(train_$QTY[i:(i+12)])
        tmp_season = append( tmp_season ,  list(value_))
        avg_tmp_season = mean(tmp_season) # average value of all tmp season values 
        avg_tmp_season = ifelse(is.na(avg_tmp_season), 1, avg_tmp_season) 
        train_season_vec = data.frame(season_ind = rep((as.numeric(tmp_season) / as.numeric(avg_tmp_season)),2))
      }
      
      training_data = cbind(train_,train_trend_vec, train_season_vec) # training data for lm model
      
      # fit MLR model
      model1 = lm(QTY ~ trend + season_ind, data = training_data[,c(2,3,4)])
      
      # if ratio (2016 / 2015) is greater than 1 the I will add trend otherwise not
      test_trend = ifelse(Ratio_2015_2016 > 1, data.frame(trend = c(25:48)) 
                          ,data.frame(trend = rep(24,24)))
      
      test_season_ind = data.frame(training_data$season_ind)
      colnames(test_season_ind) = "season_ind"

      test_trend = as.data.frame(test_trend) # transform the list into dataframe
      colnames(test_trend) = c("trend") # give name to the column
      test_data = cbind(test_trend,test_season_ind) 
      
      fitted_values = as.data.frame( fitted(model1)) # predicted values for the training data
      colnames (fitted_values) = "fitted_values"
      
      # predict 24 periods forward  
      predictions = data.frame(predict(model1, newdata = test_data))

      actual_predicted_data = data.frame( Actual = c(rep(0,48)),
                                          Predicted = c(rep(0,48)))
      
      additional_real_values_vec = c(rep(0,24))
      
      ready_actual_vec = as.data.frame(c(real_values,additional_real_values_vec))
      
      
      predicted_vec_total = c(as.vector(fitted_values),as.vector(predictions))
      actual_predicted_data["Actual"] = as.data.frame(ready_actual_vec)
      actual_predicted_data["Predicted"] = as.data.frame(predicted_vec_total)
      best_actual_predicted_data_list = append(best_actual_predicted_data_list, list(actual_predicted_data))
  }
}


# give a name to each item inside list_of_error_result 
names_2 = names(DF_subgroups_split_by_subgroup)
names(best_actual_predicted_data_list) = names_2


# ready_output is a dataframe that contains all the actual and predicted data for each degem and enables to explore the results. 
output = list()
for (i in (1:length(best_actual_predicted_data_list))){
  hierarchy = unlist(strsplit(names_2[[i]], "[.]")) # split the name of each dataframe which is of the form "11.1.1.1" to a list [11,1,1,1]
  
  subgroup_ID = data.frame(subGrp_ID = rep(hierarchy[[4]],48)) # vector of subgroup_ID's
  GroupID = data.frame(Group_ID = rep(hierarchy[[3]],48))  # vector of GroupID's
  MainGroupID = data.frame(MainGroup_ID = rep(hierarchy[[2]],48)) # vector of MainGroupID's
  MainDepartmentID = data.frame( MainDepartment_ID = rep(hierarchy[[1]],48)) # vector of MainDepartmentID's
  
  new_df = cbind(MainDepartmentID,MainGroupID,GroupID,subgroup_ID,best_actual_predicted_data_list[[i]])
  output = append(output,list(new_df))
  ready_output =  Reduce(rbind, output) # concatenate all dataframes into one dataframe
  
}




# write the output dataframe to excel
library(xlsx)
write.xlsx(ready_output, "C:/Users/heng/Desktop/first_forecast_results_Fox_subgroup_all_hierarchy_MLR???.xlsx")















