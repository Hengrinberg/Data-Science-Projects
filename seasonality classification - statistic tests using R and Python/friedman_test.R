############ friedman test for seasonality ###################

library(readxl)
install.packages("xlsx")
install.packages("rJava")
library(xlsx)
library(rJava)


number_of_sheets = c(1:3023)
sheet_vector = rep("sheet",3022)

#concatenate vectors after converting to character
sheet_names = paste0(sheet_vector,number_of_sheets)
counter = 0 # count the number of serieses that was written to excel

for (i in sheet_names ){
  DF <- read_excel("C:/Users/heng/Desktop/clean_data5.xlsx", sheet = i)
  
  # create a dataframe from the colmns: "year-month" and "pos units"
  df_for_model = DF[,c(2,12)]
  df_for_model['year__'] = as.numeric(substr(df_for_model$`Year-Month`, 1, 4))
  df_for_model['month__'] = as.numeric(substr(df_for_model$`Year-Month`,6 ,7))
  df_for_model = df_for_model[,c(2,3,4)]
  df_for_model
  
  
  start_year = as.numeric(substr(df_for_model[1,2], 1, 4))
  last_year = as.numeric(substr(df_for_model[length(df_for_model$year__),2,drop=F], 1,4))
  series_periods = last_year - start_year
  all_years_dataframe = data.frame("Jan" = numeric(0), "Feb" = numeric(0),"Mar" = numeric(0),"Apr"= numeric(0),"May" = numeric(0),
                                   "Jun" = numeric(0),"Jul" = numeric(0),"Aug" = numeric(0),"Sep" = numeric(0),"Oct" = numeric(0),
                                   "Nov" = numeric(0),"Dec" = numeric(0))
  
  # aggregate the pos units by year and month
  aggregated_series = aggregate(df_for_model$`POS Units (Net)`~df_for_model$year__ +df_for_model$month__, data=df_for_model, sum)
  
  colnames(aggregated_series) <- c("year","month","POS Units (Net)") # change column names
  ranks = with(aggregated_series, order(aggregated_series$year, aggregated_series$month)) # sort the aggregated dataframe by year and then by month
  aggregated_series[ranks,]
 
  
    
  for (series in 0: series_periods){

    # creation of separate vector for each year
    vector_of_months = list(1,2,3,4,5,6,7,8,9,10,11,12)
    year_data = aggregated_series[aggregated_series$year == start_year + series,]
    
  
    
    year_sales_vector = c(year_data$`POS Units (Net)`)
    names(year_sales_vector) = year_data$month # giving each value a name which indicates his month
    

    diff_vector = setdiff(vector_of_months,as.numeric(names(year_sales_vector))) # create difference vector to replace with the median value
    vector_of_medians = rep(median(year_sales_vector),length(diff_vector)) # craete vector with the median value repeated n times depend on the number of missing observations
    names(vector_of_medians) = diff_vector # giving each value a name which indicates his month
    
  
    all_year_data = c(vector_of_medians,year_sales_vector) # vector that contains the medians vector and the given sales   
    unsorted_vector = as.numeric(names(all_year_data)) # store all months as numeric type in a vector
    rankings = order(unsorted_vector) # orderd vector from 1 to 12
    sorted_vector = as.vector(matrix(all_year_data)[rankings,])
    
    # give name to each observation in the vector (each name indicates the month of the observation)
    names(sorted_vector) = list("Jan", "Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
    
    
    # append the vector to the general years dataframe
    all_years_dataframe = rbind(all_years_dataframe, sorted_vector)
    
    
    #give each column in the dataframa the corresponding month as name
    colnames(all_years_dataframe) = list("Jan", "Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
    
  }        

  # transform the dataframe to matrix 
  matrix_ = as.matrix(all_years_dataframe)
  
  if(nrow(matrix_) > 1){ # check if we have more than one year, if not the test has no meaning
  
  # friedman's test
  result = friedman.test(matrix_)$p.value
  
  # test if the result (seasonality) is significant
  Is_seasonal = ifelse(result <= 0.05, "true", "false")
  
  #create a vector with the test result
  friedman_test_result = rep(Is_seasonal,length(DF$`Year-Month`))
  friedman_pvalue_result = rep(result,length(DF$`Year-Month`))
  
  #append the "friedman_test_result(monthly)" to the original series
  DF["friedman_test_result(monthly)"] = friedman_test_result
  DF["friedman_test_pvalue"] = friedman_pvalue_result
  
  counter = counter + 1
  print(counter)
  # write each dataframe into seperate excel spreadsheet
  write.xlsx(DF, file = "C:/Users/heng/Desktop/write_test_file8.xlsx",sheetName = i, append = TRUE)
  
  }
}






