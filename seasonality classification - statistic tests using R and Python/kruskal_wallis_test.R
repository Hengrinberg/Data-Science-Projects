############ kruskal-wallis test for seasonality ###################


install.packages("readxl")
library(readxl)
number_of_sheets = c(50:100) #c(1:3023)
sheet_vector = rep("sheet",51)  #rep("sheet",3022)

#concatenate vectors after converting to character
sheet_names = paste0(sheet_vector,number_of_sheets)

list_of_dataframes = list()
for (i in sheet_names ){
  DF <- read_excel("C:/Users/Home Premium/Desktop/clean_data5.xlsx", sheet = i)
  
  
  # create a dataframe from the colmns: "year-month" and "pos units"
  df_for_model = DF[,c(2,12)]
  
  
  # aggregate the pos units by year and month
  aggregated_series = aggregate(df_for_model$`POS Units (Net)`, by=list(Category=df_for_model$`Year-Month` ), FUN=sum)
  aggregated_series["Month"] = as.numeric(substr(aggregated_series[,1], 6,7)) # add month column to aggregated_series
  
  
  # create a vector of pos unit values for each month
  months_list = list(
    Jan = c(aggregated_series$x [aggregated_series$Month == 1]),
    Feb = c(aggregated_series$x [aggregated_series$Month == 2]),
    Mar = c(aggregated_series$x [aggregated_series$Month == 3]),
    Apr = c(aggregated_series$x [aggregated_series$Month == 4]),
    May = c(aggregated_series$x [aggregated_series$Month == 5]),
    Jun = c(aggregated_series$x [aggregated_series$Month == 6]),
    Jul = c(aggregated_series$x [aggregated_series$Month == 7]),
    Aug = c(aggregated_series$x [aggregated_series$Month == 8]),
    Sep = c(aggregated_series$x [aggregated_series$Month == 9]),
    Oct = c(aggregated_series$x [aggregated_series$Month == 10]),
    Nov = c(aggregated_series$x [aggregated_series$Month == 11]),
    Dec = c(aggregated_series$x [aggregated_series$Month == 12]))
  months_list
  
  
  # find the length of the month with the largest number of observations
  longest_vector = max(length(months_list$Jan),length(months_list$Feb),length(months_list$Mar),length(months_list$Apr),
                       length(months_list$May),length(months_list$Jun),length(months_list$Jul),length(months_list$Aug)
                       ,length(months_list$Sep),length(months_list$Oct),length(months_list$Nov),length(months_list$Dec))
  
  
  # if there are no observations in the vector - 0 will be written instead 
  for (j in (1:length(months_list))){
    if (length(months_list[[j]]) == 0){months_list[[j]] = c(rep(0, longest_vector))}
    else{
      # craete seperate vectors for each month and append the median of the vector instead of the missing observations
      months_list[[j]] = c(months_list[[j]], rep(median(months_list[[j]]), longest_vector - length(months_list[[j]])))}
  }
  
  # create one dataframe from all the vectors
  ready_dataframe = data.frame(cbind(months_list$Jan, months_list$Feb ,months_list$Mar,
                                        months_list$Apr, months_list$May, months_list$Jun, months_list$Jul, months_list$Aug, months_list$Sep, 
                                        months_list$Oct, months_list$Nov, months_list$Dec))
  
  # kruskal-wallis test
  result = kruskal.test(ready_dataframe)$p.value
   
  counter = 0
  # test if the result (seasonality) is significant
  Is_seasonal = ifelse(result <= 0.05, "true", "false")
  
  
  #create a vector with the test result
  kruskal_wallis_test_result = rep(Is_seasonal,length(DF$`Year-Month`))
  kruskal_wallis_test_pvalue = rep(result,length(DF$`Year-Month`))
  #append the "kruskal_wallis_test_result(monthly)" to the original series
  DF["kruskal_wallis_test_result(monthly)"] = kruskal_wallis_test_result
  DF["p-value"] = kruskal_wallis_test_pvalue
  
  list_of_dataframes = append(list_of_dataframes,list(DF))
  
  
  #install.packages("xlsx")
  #library(xlsx)
  #Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk1.8.0_144/")
  #Sys.getenv("JAVA_HOME")
  #install.packages("rJava")
  #library(rJava)
  
  # write each dataframe into seperate excel spreadsheet
  #write.xlsx(DF, file = "C:/Users/Home Premium/Desktop/write_test_file2.xlsx",sheetName = i, append = TRUE)
  
}
