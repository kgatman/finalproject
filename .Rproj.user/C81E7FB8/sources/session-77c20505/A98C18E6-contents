#install.packages("/Users/kgatman/Downloads/h2o-3.46.0.8/R/h2o_3.46.0.8.tar.gz")

library(h2o)

#Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk-1.8.jdk/")

h2o.init()


# loading additional packages for preprocessing

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting train / test sets
library(caret) # performance metric functions
library(pROC) # used for obtaining AUC
#install.packages("tidyverse")
library(tidyverse)

trace_data <- read_csv("TRACE_Data.csv")

trace_data <- na.omit(trace_data)
traceData <- as.data.frame(trace_data)
str(traceData)

# to avoid multicolinearity
# I decided to remove the following variables

traceData_final <- traceData %>%
  select(
    - `Organisation unit name`,
    - `HIV testing point`,
  )

# renaming my variables for better code

traceData_final <- traceData_final %>%
  rename(
    orgUnit = `Organisation unit code`,
    marital_status = `marital status`,
    recency_interpretation = `recency Interpretation`,
    test_name = `test name`,
    marriage_age = `marriage age`,
    age = `participant age`,
    pregnancy_status = `pregnancy status`
  )

traceData_final <- traceData_final %>%
  select(
    -test_name
  )


#### setting categorical variables
traceData_final$District <- factor(traceData_final$District, ordered = FALSE)
traceData_final$orgUnit <- factor(traceData_final$orgUnit, ordered = FALSE)


### simplifying the Occupation variable to be just Working/Not Working

### transforming and simplifying the occupation variable

traceData_final <- traceData_final %>%
  mutate(
    Occupation_simple = case_when(
      occupation %in% c(
        "Domestic Worker",
        "Farmer",
        "Migrant worker",
        "Student",
        "Vendor",
        "Other"
      ) ~ "Working",
      
      occupation %in% c(
        "Not working",
        "Refused to answer"
      ) ~ "Not Working",
      
      TRUE ~ "Not working"   # catch unclassified entries
    )
  )

traceData_final <- traceData_final %>%
  select(
    -occupation
  )



### transforming and simplifying the marital_status variable

traceData_final <- traceData_final %>%
  mutate(
    married_simple = case_when(
      marital_status %in% c(
        "Married",
        "Cohabiting",
        "Refused to answer"
      ) ~ "Married",
      
      marital_status %in% c(
        "Divorced",
        "Widowed",
        "Single"
      ) ~ "Not_married",
      
      TRUE ~ "Not_married"   # catch unclassified entries
    )
  )

traceData_final <- traceData_final %>%
  select(
    -marital_status
  )




### transforming and simplifying the marital_status variable

traceData_final <- traceData_final %>%
  mutate(
    pregnant_simple = case_when(
      pregnancy_status %in% c(
        "No",
        "Not applicable"
      ) ~ "Not_pregnant",
      
      pregnancy_status %in% c(
        "Yes"
      ) ~ "Pregnant",
      
      TRUE ~ "Not_pregnant"   # catch unclassified entries
    )
  )

traceData_final <- traceData_final %>%
  select(
    -pregnancy_status
  )


traceData_final$District <- factor(traceData_final$District, ordered = FALSE)
traceData_final$orgUnit <- factor(traceData_final$orgUnit, ordered = FALSE)
traceData_final$Occupation_simple <- factor(traceData_final$Occupation_simple, ordered = FALSE)
traceData_final$gender <- factor(traceData_final$gender, ordered = FALSE)
traceData_final$married_simple <- factor(traceData_final$married_simple, ordered = FALSE)
traceData_final$pregnant_simple <- factor(traceData_final$pregnant_simple, ordered = FALSE)
traceData_final$recency_interpretation <- factor(traceData_final$recency_interpretation, ordered = FALSE)

#####

traceData_ready <- traceData_final %>%
  filter(recency_interpretation != "Negative")
traceData_ready <- traceData_final %>%
  filter(recency_interpretation != "Negative")
traceData_ready <- traceData_final %>%
  filter(recency_interpretation != "Invalid")

recency_col <- grep("^recency|Recency", names(traceData_ready), value = TRUE)

traceData_ready <- traceData_ready %>%
  filter(!(.data[[recency_col]] == "Invalid"))

traceData_ready$recency_interpretation <- factor(traceData_ready$recency_interpretation, ordered = FALSE)

summary(traceData_ready)

prop.table(table(traceData_ready$recency_interpretation))


traceData_ready$recency_interpretation <- ifelse(
  traceData_ready$recency_interpretation == "Recent", 1L, 0L)

prop.table(table(traceData_ready$recency_interpretation))

#Discarding other levels that shouldn't have been here

traceData_final <- traceData_final[traceData_final$recency_interpretation!="Invalid",]
traceData_final <- traceData_final[traceData_final$recency_interpretation!="Negative",]

# discard lingering levels
traceData_final$recency_interpretation <- droplevels(traceData_final$recency_interpretation)