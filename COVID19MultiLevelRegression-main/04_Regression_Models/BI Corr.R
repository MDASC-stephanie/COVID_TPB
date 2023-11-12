BI_corr <- read.csv('lv2 corr.csv', header = TRUE, sep = ',')


cols <- c("A", "SN", "PBC", "O", "BI")
countries = c("Australia", "Canada", "United Kingdom", "United States")


library(corrr)
library(dplyr)

for (c in countries){
  print(paste("******************************", c, "***********************************"))
  #data of that country
  data <- subset(BI_corr, country==c, select = cols)

  print(data %>% correlate(quiet = TRUE) %>% focus(BI))
  #For reference: with significance test on the correlation coefficient
  #print(data %>% cor_test(vars = "BI", vars2 = c("A", "SN", "PBC", "O")))
  
}