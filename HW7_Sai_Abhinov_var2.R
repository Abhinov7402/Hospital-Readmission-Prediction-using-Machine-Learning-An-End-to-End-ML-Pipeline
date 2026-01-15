library(dplyr)
library(readr)
library(ggplot2)
library(lattice)
library(caret)


# Load the data
TrainData <- read_csv("C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/hm7-Train-2024.csv")
TestData <- read_csv("C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/hm7-Test-2024.csv")

# Remove rows with missing patientID in TestData
TestData <- TestData[!is.na(TestData$patientID), ]

# Fill missing numeric values in TrainData with the median
Train_numeric_columns <- sapply(TrainData, is.numeric)
TrainData[Train_numeric_columns] <- lapply(TrainData[Train_numeric_columns], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
})

# Fill missing character values in TrainData with "Unknown"
Train_categorical_columns <- sapply(TrainData, is.character)
TrainData[Train_categorical_columns] <- lapply(TrainData[Train_categorical_columns], function(x) {
  x[is.na(x)] <- "Unknown"
  return(x)
})

# Remove columns if they exist
cols_to_remove <- c("examide", "citoglipton", "glimepiride.pioglitazone")
TrainData <- TrainData %>%
  select(any_of(setdiff(names(TrainData), cols_to_remove)))
TestData <- TestData %>%
  select(any_of(setdiff(names(TestData), cols_to_remove)))

# Process Test Data similarly
Test_numeric_columns <- sapply(TestData, is.numeric)
TestData[Test_numeric_columns] <- lapply(TestData[Test_numeric_columns], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
})

Test_categorical_columns <- sapply(TestData, is.character)
TestData[Test_categorical_columns] <- lapply(TestData[Test_categorical_columns], function(x) {
  x[is.na(x)] <- "Unknown"
  return(x)
})


# Convert target variable 'readmitted' to a factor for classification
TrainData$readmitted <- as.factor(TrainData$readmitted)
TrainData[sapply(TrainData, is.character)] <- lapply(TrainData[sapply(TrainData, is.character)], as.factor)
TestData[sapply(TestData, is.character)] <- lapply(TestData[sapply(TestData, is.character)], as.factor)

# Define age group conversion function and apply it
convert_age <- function(age) {
  if (age %in% c("[0-10)", "[10-20)", "[20-30)")) {
    return("Young")
  } else if (age %in% c("[30-40)", "[40-50)", "[50-60)")) {
    return("Middle-aged")
  } else {
    return("Old")
  }
}
# converting age factor to three levels
TrainData$age <- factor(sapply(TrainData$age, convert_age), levels = c("Young", "Middle-aged", "Old"))
TestData$age <- factor(sapply(TestData$age, convert_age), levels = c("Young", "Middle-aged", "Old"))

# converting admission_type from numeric to factor
TrainData$admission_type <- factor(TrainData$admission_type, levels = c("1", "2", "3","4", "5", "6", "7","8"))
TestData$admission_type <- factor(TestData$admission_type, levels = c("1", "2", "3","4", "5", "6", "7","8"))

# converting admission_source from numeric to factor
TrainData$admission_source <- factor(TrainData$admission_source, levels = c("1", "2", "3","4", "5", "6", "7","8","9", "10", "11", "12", "13", "14", "15"," 16", "17", "18","19","20","21","22", "23", "24", "25", "26"))
TestData$admission_source <- factor(TestData$admission_source, levels = c("1", "2", "3","4", "5", "6", "7","8","9", "10", "11", "12", "13", "14", "15"," 16", "17", "18","19","20","21","22", "23", "24", "25", "26"))





# Convert 'readmitted' to a binary factor with levels "yes" and "no"
TrainData$readmitted <- factor(ifelse(TrainData$readmitted == 1, "yes", "no"), levels = c("no", "yes"))

# function get_mode() to handle categorical missing value imputation
get_mode <- function(x) {
  uniq_vals <- unique(na.omit(x))
  uniq_vals[which.max(tabulate(match(x, uniq_vals)))]
}
str(TrainData)
str(TestData)

# Aggregate TrainData by patientID for further modeling
TrainData_agg <- TrainData %>%
  group_by(patientID) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE),
            across(where(is.factor), ~ get_mode(.x)))
TestData_agg <- TestData %>%
  group_by(patientID) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE),
            across(where(is.factor), ~ get_mode(.x)))

str(TrainData_agg)
str(TestData_agg)

# Impute missing values in numeric columns using the median
TrainData_agg[sapply(TrainData_agg, is.numeric)] <- lapply(TrainData_agg[sapply(TrainData_agg, is.numeric)], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)  # Impute with the median
  return(x)
})

TestData_agg[sapply(TestData_agg, is.numeric)] <- lapply(TestData_agg[sapply(TestData_agg, is.numeric)], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)  # Impute with the median
  return(x)
})
######################################
# Ensure 'readmitted' column has no missing values
sum(is.na(TrainData_agg$readmitted))  # Check for missing values in target column

# If 'readmitted' is missing, we should impute or remove those rows (remove as last resort)
TrainData_agg$readmitted[is.na(TrainData_agg$readmitted)] <- "No"  # Replace NAs with "No"

# Check again to ensure no missing values
sum(is.na(TrainData_agg$readmitted))

# Impute missing values for other columns as previously
TrainData_agg[sapply(TrainData_agg, is.numeric)] <- lapply(TrainData_agg[sapply(TrainData_agg, is.numeric)], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)  # Impute with the median
  return(x)
})

TestData_agg[sapply(TestData_agg, is.numeric)] <- lapply(TestData_agg[sapply(TestData_agg, is.numeric)], function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)  # Impute with the median
  return(x)
})

# Impute missing values for categorical columns using the mode
get_mode <- function(x) {
  uniq_vals <- unique(na.omit(x))
  uniq_vals[which.max(tabulate(match(x, uniq_vals)))]
}

TrainData_agg[sapply(TrainData_agg, is.factor)] <- lapply(TrainData_agg[sapply(TrainData_agg, is.factor)], function(x) {
  x[is.na(x)] <- get_mode(x)  # Impute with the mode
  return(x)
})
# Impute missing values for categorical columns using the mode
get_mode <- function(x) {
  uniq_vals <- unique(na.omit(x))
  uniq_vals[which.max(tabulate(match(x, uniq_vals)))]
}

TestData_agg[sapply(TestData_agg, is.factor)] <- lapply(TestData_agg[sapply(TestData_agg, is.factor)], function(x) {
  x[is.na(x)] <- get_mode(x)  # Impute with the mode
  return(x)
})


TrainData_agg$discharge_disposition <- factor(TrainData_agg$discharge_disposition, levels = c("1", "2", "3","4", "5", "6", "7","8","9", "10", "11", "12", "13", "14", "15"," 16", "17", "18","19","20","21","22", "23", "24", "25", "26", "27", "28","29"))
TestData_agg$discharge_disposition <- factor(TestData_agg$discharge_disposition, levels = c("1", "2", "3","4", "5", "6", "7","8","9", "10", "11", "12", "13", "14", "15"," 16", "17", "18","19","20","21","22", "23", "24", "25", "26", "27", "28","29"))

# Impute missing values with the level "26"
TrainData_agg$discharge_disposition[is.na(TrainData_agg$discharge_disposition)] <- "26"
# Impute missing values with the level "26"
TestData_agg$discharge_disposition[is.na(TestData_agg$discharge_disposition)] <- "26"

colSums(is.na(TrainData_agg))  # Check how many NAs are in each column
colSums(is.na(TestData_agg))  # Check how many NAs are in each column


head(TrainData_agg)
head(TestData_agg)

#####################################################################################################
#
#           LOGISTIC REGRESSION
#
################################################################################################
library(dplyr)
library(MLmetrics)  # For LogLoss metric

# Define the formula for logistic regression, excluding 'patientID' as it's an identifier
#formula <- readmitted ~ time_in_hospital + num_lab_procedures + num_procedures + indicator_level +
#  age + admission_type + discharge_disposition + number_diagnoses + admission_source + number_outpatient + number_inpatient
#formula <- readmitted ~ admission_type * age
formula <- readmitted ~  time_in_hospital + num_lab_procedures + num_procedures + indicator_level+ 
  indicator_2_level+age+gender+ num_medications+ 
  admission_type + discharge_disposition + race  + number_diagnoses +  
  admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
  payer_code + number_outpatient + number_inpatient +number_emergency + metformin 

#formula <- readmitted ~  time_in_hospital + num_lab_procedures + num_procedures + indicator_level+ 
#  indicator_2_level+age+gender+ num_medications+ 
#  admission_type + discharge_disposition + race  + number_diagnoses + num_procedures + 
#  admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
#  payer_code + number_outpatient + number_inpatient +number_emergency + metformin +
#  repaglinide + nateglinide + chlorpropamide + glimepiride + acetohexamide + glipizide +
#  glyburide + tolbutamide + pioglitazone + rosiglitazone + acarbose + miglitol + troglitazone +
#  tolazamide + glyburide-metformin + glipizide-metformin + metformin-rosiglitazone + metformin-pioglitazone 

formula <- readmitted ~     indicator_level+ age+ gender +race + admission_type +
  indicator_2_level+ num_medications + discharge_disposition  + number_diagnoses +
    admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
  payer_code + number_outpatient + number_inpatient  + number_emergency + metformin 

  


# Define a custom summary function to include LogLoss, Accuracy, and Kappa
customSummary <- function(data, lev = NULL, model = NULL) {
  
  # Extract predicted probabilities for the "yes" class
  prob_yes <- data$yes
  
  # Convert probabilities to binary predictions with a 0.5 threshold
  pred_class <- factor(ifelse(prob_yes > 0.5, "yes", "no"), levels = c("no", "yes"))
  
  # Calculate Accuracy
  accuracy <- mean(pred_class == data$obs)
  
  # Calculate Kappa
  kappa <- confusionMatrix(pred_class, data$obs)$overall["Kappa"]
  
  # Calculate LogLoss
  logloss <- LogLoss(prob_yes, ifelse(data$obs == "yes", 1, 0))
  
  # Return all metrics in a named vector
  out <- c(Accuracy = accuracy, Kappa = kappa, LogLoss = logloss)
  return(out)
}



# Set up cross-validation control
trControl <- trainControl(
  method = "cv",               # Use cross-validation
  number = 5,                  # Number of folds
  classProbs = TRUE,           # For probability predictions
  summaryFunction = customSummary # Custom summary function
)

# Train the logistic regression model
log_reg_model <- train(
  formula, 
  data = TrainData_agg, 
  method = "glm", 
  family = "binomial",
  trControl = trControl
)

# Print model summary
print(log_reg_model)

# Predict probabilities on the test data
log_reg_pred_test <- predict(log_reg_model, TestData_agg, type = "prob")
print(log_reg_pred_test)

# Extract probabilities for the "Yes" class (readmission probability)
readmission_probabilities <- log_reg_pred_test$yes
print(readmission_probabilities)

# Create a dataframe with patientID and the predicted readmission probability
submission <- data.frame(patientID = TestData_agg$patientID, predReadmit = readmission_probabilities)

# Check the number of rows in the submission to ensure it matches TestData_agg
nrow(submission) == nrow(TestData_agg)

# Write the predictions to a CSV file
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/Logistic_readmission_predictbestshortm.csv", row.names = FALSE)




############################################################################

#  decision tree

####################################################
# Convert the target variable to factor for classification
TrainData$readmitted <- as.factor(TrainData$readmitted)
TrainData[sapply(TrainData, is.character)] <- lapply(TrainData[sapply(TrainData, is.character)], as.factor)


# Load required packages
library(caret)
library(MLmetrics)

# Define a custom summary function to include LogLoss, Accuracy, and Kappa
customSummary <- function(data, lev = NULL, model = NULL) {
  
  # Extract predicted probabilities for the "yes" class
  prob_yes <- data$yes
  
  # Convert probabilities to binary predictions with a 0.5 threshold
  pred_class <- factor(ifelse(prob_yes > 0.5, "yes", "no"), levels = c("no", "yes"))
  
  # Calculate Accuracy
  accuracy <- mean(pred_class == data$obs)
  
  # Calculate Kappa
  kappa <- confusionMatrix(pred_class, data$obs)$overall["Kappa"]
  
  # Calculate LogLoss
  logloss <- LogLoss(prob_yes, ifelse(data$obs == "yes", 1, 0))
  
  # Return all metrics in a named vector
  out <- c(Accuracy = accuracy, Kappa = kappa, LogLoss = logloss)
  return(out)
}
# Use the custom summary function in trainControl
trControl <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = customSummary
)


# Specify the formula for the model (include relevant predictors) Deci tree
formula <- readmitted ~  time_in_hospital + num_lab_procedures + num_procedures + indicator_level+ 
  indicator_2_level+age+gender+  
 admission_type + discharge_disposition + race  + number_diagnoses + num_procedures + 
  admission_source + insulin +  diabetesMed + race + A1Cresult + max_glu_serum+
  payer_code + number_outpatient + number_inpatient +number_emergency + metformin
#formula <- readmitted ~ time_in_hospital + num_lab_procedures + num_procedures + indicator_level +
 # age + admission_type + discharge_disposition + number_diagnoses + admission_source + number_outpatient + number_inpatient+ insulin+
  #number_emergency+ diabetesMed + race + A1Cresult + max_glu_serum

#try below for 
formula <- readmitted ~     indicator_level+ age+ gender +race + admission_type +
  indicator_2_level+ num_medications + discharge_disposition  + number_diagnoses +
  admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
  payer_code + number_outpatient + number_inpatient  + number_emergency + metformin 


# Decision Tree Model using rpart
tree_model <- train(
  formula, 
  data = TrainData, 
  method = "rpart",  # Use decision tree method
  trControl = trControl 
)
#trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = mnLogLoss)

summary(tree_model)
# Print cross-validation accuracy and Kappa
print(tree_model$results)





# Visualize the decision tree (optional)
# rpart.plot(tree_model$finalModel)  # Uncomment this line to plot the decision tree

# Make predictions on the TestData
tree_pred_test <- predict(tree_model, TestData, type = "prob")

# Get the probability of 'Yes' (readmitted) for each test case
readmission_probabilities <- tree_pred_test$yes

# Prepare the submission dataframe
submission <- data.frame(patientID = TestData$patientID, predReadmit = readmission_probabilities)

# Check for extra patientIDs (those present in submission but not in TestData)
extra_patientIDs <- setdiff(submission$patientID, TestData$patientID)
print(extra_patientIDs)

# Write the submission data to CSV
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/DecisionTree_submissionNEWVAR2_bestshot.csv", row.names = FALSE)




#####################################################################################################
#
#          Penalized LOGISTIC REGRESSION (Hyper parameter tuning)
#
################################################################################################
library(dplyr)
library(MLmetrics)  # For LogLoss metric

# Define the formula for logistic regression, excluding 'patientID' as it's an identifier
#formula <- readmitted ~ time_in_hospital + num_lab_procedures + num_procedures + indicator_level +
#  age + admission_type + discharge_disposition + number_diagnoses + admission_source + number_outpatient + number_inpatient
#formula <- readmitted ~ admission_type * age
formula <- readmitted ~  time_in_hospital + num_lab_procedures + num_procedures + indicator_level+ 
  indicator_2_level+age+gender+ num_medications+ 
  admission_type + discharge_disposition + race  + number_diagnoses +  
  admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
  payer_code + number_outpatient + number_inpatient +number_emergency + metformin 

#formula <- readmitted ~  time_in_hospital + num_lab_procedures + num_procedures + indicator_level+ 
#  indicator_2_level+age+gender+ num_medications+ 
#  admission_type + discharge_disposition + race  + number_diagnoses + num_procedures + 
#  admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
#  payer_code + number_outpatient + number_inpatient +number_emergency + metformin +
#  repaglinide + nateglinide + chlorpropamide + glimepiride + acetohexamide + glipizide +
#  glyburide + tolbutamide + pioglitazone + rosiglitazone + acarbose + miglitol + troglitazone +
#  tolazamide + glyburide-metformin + glipizide-metformin + metformin-rosiglitazone + metformin-pioglitazone 

formula <- readmitted ~     indicator_level+ age+ gender +race + admission_type +
  indicator_2_level+ num_medications + discharge_disposition  + number_diagnoses +
  admission_source + insulin +  diabetesMed  + A1Cresult + max_glu_serum+
  payer_code + number_outpatient + number_inpatient  + number_emergency + metformin 




# Define a custom summary function to include LogLoss, Accuracy, and Kappa
customSummary <- function(data, lev = NULL, model = NULL) {
  
  # Extract predicted probabilities for the "yes" class
  prob_yes <- data$yes
  
  # Convert probabilities to binary predictions with a 0.5 threshold
  pred_class <- factor(ifelse(prob_yes > 0.5, "yes", "no"), levels = c("no", "yes"))
  
  # Calculate Accuracy
  accuracy <- mean(pred_class == data$obs)
  
  # Calculate Kappa
  kappa <- confusionMatrix(pred_class, data$obs)$overall["Kappa"]
  
  # Calculate LogLoss
  logloss <- LogLoss(prob_yes, ifelse(data$obs == "yes", 1, 0))
  
  # Return all metrics in a named vector
  out <- c(Accuracy = accuracy, Kappa = kappa, LogLoss = logloss)
  return(out)
}

# Define tuning grid for alpha and lambda
# Alpha: 0 for Ridge, 1 for Lasso, values in between for Elastic Net
tuneGrid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),  # Adjust this range as needed
  lambda = 10^seq(-3, 1, length = 10)  # Lambda values from small to large
)

# Set up cross-validation control
trControl <- trainControl(
  method = "cv",               # Use cross-validation
  number = 5,                  # Number of folds
  classProbs = TRUE,           # For probability predictions
  summaryFunction = customSummary # Custom summary function
)

# Train the logistic regression model
log_reg_model <- train(
  formula, 
  data = TrainData_agg, 
  method = "glm", 
  family = "binomial",
  trControl = trControl,
  tuneGrid = tuneGrid,
  metric = "LogLoss"
)

# Print model summary
print(log_reg_model$bestTune)

print(log_reg_model)

# Predict probabilities on the test data
log_reg_pred_test <- predict(log_reg_model, TestData_agg, type = "prob")
print(log_reg_pred_test)

# Extract probabilities for the "Yes" class (readmission probability)
readmission_probabilities <- log_reg_pred_test$yes
print(readmission_probabilities)

# Create a dataframe with patientID and the predicted readmission probability
submission <- data.frame(patientID = TestData_agg$patientID, predReadmit = readmission_probabilities)

# Check the number of rows in the submission to ensure it matches TestData_agg
nrow(submission) == nrow(TestData_agg)

# Write the predictions to a CSV file
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/Logistic_readmission_predictbestshortm.csv", row.names = FALSE)


#######################################################################################################
#
#                     RANDOM FOREST
#
######################################################################################

# Load required packages
library(caret)
library(randomForest)
install.packages("ModelMetrics")  # Install if you haven't already
library(ModelMetrics)

# Define a custom LogLoss function
LogLoss <- function(actual, predicted) {
  epsilon <- 1e-15  # To avoid log(0)
  predicted <- pmax(pmin(predicted, 1 - epsilon), epsilon)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

# Define the formula
formula <- readmitted ~ indicator_level + age + gender + race + admission_type +
  indicator_2_level + num_medications + discharge_disposition + number_diagnoses +
  admission_source + insulin + diabetesMed + A1Cresult + max_glu_serum +
  payer_code + number_outpatient + number_inpatient + number_emergency + metformin 

# Define a custom summary function to include LogLoss, Accuracy, and Kappa
customSummary <- function(data, lev = NULL, model = NULL) {
  
  # Extract predicted probabilities for the "yes" class
  prob_yes <- data$yes
  
  # Convert probabilities to binary predictions with a 0.5 threshold
  pred_class <- factor(ifelse(prob_yes > 0.5, "yes", "no"), levels = c("no", "yes"))
  
  # Ensure data$obs is a factor with the same levels as pred_class
  data$obs <- factor(data$obs, levels = c("no", "yes"))
  
  # Calculate Accuracy
  accuracy <- mean(pred_class == data$obs)
  
  # Calculate Kappa with an additional check to prevent errors
  kappa_result <- tryCatch({
    confusionMatrix(pred_class, data$obs)$overall["Kappa"]
  }, error = function(e) {
    NA  # Return NA if there's an error
  })
  
  # Calculate LogLoss
  logloss <- LogLoss(ifelse(data$obs == "yes", 1, 0), prob_yes)
  
  # Return all metrics in a named vector
  out <- c(Accuracy = accuracy, Kappa = kappa_result, LogLoss = logloss)
  return(out)
}

# Set up cross-validation control
trControl <- trainControl(
  method = "cv",               # Use cross-validation
  number = 5,                  # Number of folds
  classProbs = TRUE,           # For probability predictions
  summaryFunction = customSummary # Custom summary function
)

# Define the tuning grid for mtry (number of variables randomly sampled as candidates at each split)
tuneGrid <- expand.grid(mtry = c(1, 2, 5, 9, 10, 11, 12))  # Customize mtry values as needed


# Train the random forest model
rf_model <- train(
  formula, 
  data = TrainData_agg, 
  method = "rf",
  trControl = trControl,
  tuneGrid = tuneGrid,
  metric = "LogLoss"  # Optimize for LogLoss
)

# Print the best tuning parameter and model summary
print(rf_model$bestTune)
print(rf_model)

# Predict probabilities on the test data
rf_pred_test <- predict(rf_model, TestData_agg, type = "prob")
print(rf_pred_test)

# Extract probabilities for the "yes" class (readmission probability)
readmission_probabilities <- rf_pred_test$yes
print(readmission_probabilities)

# Create a dataframe with patientID and the predicted readmission probability
submission <- data.frame(patientID = TestData_agg$patientID, predReadmit = readmission_probabilities)

# Check the number of rows in the submission to ensure it matches TestData_agg
nrow(submission) == nrow(TestData_agg)

# Write the predictions to a CSV file
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/RandomForest_readmission_predictvar2bestshot12.csv", row.names = FALSE)


#########################################################################################################

################# Neural Nets
# Convert `readmitted` to a factor in `TrainData_agg`
TrainData_agg$readmitted <- as.factor(TrainData_agg$readmitted)
# Load necessary libraries
library(nnet)
library(caret)
library(MLmetrics)
# Define cross-validation settings with accuracy and kappa as metrics
train_control <- trainControl(
  method = "cv",             # Cross-validation method
  number = 3,               # Number of folds
  classProbs = TRUE,         # Needed for class probabilities
  summaryFunction = twoClassSummary  # For accuracy and kappa
)

# Train the model using caret with cross-validation
nn_model_cv <- train(
  formula = formula,
  data = TrainData_agg,
  method = "nnet",
  trControl = train_control,
  linout = FALSE,            # FALSE because this is a classification
  maxit = 100,
  trace = TRUE,
  metric = "Accuracy",       # Use accuracy as the primary metric
  tuneGrid = expand.grid(size = 1, decay = 0.1)  # Adjust grid as needed
)

# View cross-validation results for accuracy and kappa
print(nn_model_cv)

# Access accuracy and kappa from the cross-validation results
accuracy <- nn_model_cv$results$Accuracy[1]
kappa <- nn_model_cv$results$Kappa[1]
print(paste("Cross-validated Accuracy:", accuracy))
print(paste("Cross-validated Kappa:", kappa))

# Predict probabilities on the test set for "Yes" class
nn_pred_test <- predict(nn_model_cv, TestData_agg, type = "prob")[, "Yes"]

# Prepare submission
submission <- data.frame(patientID = TestData_agg$patientID, predReadmit = nn_pred_test)
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/submission_neural_net_best_63.csv", row.names = FALSE)


library(nnet)
library(MLmetrics)

# Define the formula
#formula <- readmitted ~ time_in_hospital + num_lab_procedures + age + admission_type + diabetesMed 
formula <- readmitted ~ indicator_level + admission_type + age+ gender +race +
  indicator_2_level+ num_medications + 
  admission_source + 
  payer_code + number_outpatient + number_inpatient  + number_emergency  + metformin

#insulin +  diabetesMed  + A1Cresult + max_glu_serum+  + metformin  + age+ gender +race  number_diagnoses + discharge_disposition  + 

# Train a neural network model
nn_model <- nnet(
  formula = formula, 
  data = TrainData_agg, 
  size = 1,      # Number of hidden neurons 
  linout = FALSE,  # FALSE because we are performing classification
  maxit = 100,   # Maximum iterations for training
  trace = TRUE   # Show the training progress
)

# Predict probabilities for the training data
nn_pred_train <- predict(nn_model, TrainData_agg, type = "raw")  # Raw outputs from the neural network (probabilities)

# Predict probabilities for the test data
nn_pred_test <- predict(nn_model, TestData_agg, type = "raw")  # Raw outputs (probabilities)

# Inspect the structure of the predictions
str(nn_pred_train)

# Create actual labels (convert to numeric for LogLoss calculation)
actual_train_labels <- ifelse(TrainData_agg$readmitted == "Yes", 1, 0)


# Calculate Log Loss for training data
log_loss_train <- LogLoss(actual_train_labels, nn_pred_train)
print(paste("Log Loss (Train):", log_loss_train))

# Prepare submission (for test set)
submission <- data.frame(patientID = TestData_agg$patientID, predReadmit = nn_pred_test)
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/submission_neural_net_best_63.csv", row.names = FALSE)


##############################################################################################################
#
#                 KNN
#
########################################################################################################3
library(class)
# k-Nearest Neighbors (kNN)
formula <- readmitted ~ indicator_level + admission_type + age+ gender +race +
  indicator_2_level+ num_medications + 
  admission_source + 
  payer_code + number_outpatient + number_inpatient  + number_emergency  + metformin


knn_model <- train(
  formula, 
  data = TrainData_agg, 
  method = "knn", 
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = mnLogLoss)
)

knn_pred_train <- predict(knn_model, TrainData_agg)
conf_matrix_knn <- confusionMatrix(knn_pred_train, TrainData$readmitted)
print(conf_matrix_knn)

knn_pred_test <- predict(knn_model, TestData_agg, type = "prob")
readmission_probabilities_knn <- knn_pred_test$Yes

submission_knn <- data.frame(patientID = TestData$patientID, predReadmit = readmission_probabilities_knn)
write.csv(submission_knn, "knn_submission.csv", row.names = FALSE)

#######################################################################################################################

#     MARS 

##############################################################################################

library(caret)
library(earth)
library(MLmetrics)

# The formula for the MARS model
formula <- readmitted ~ indicator_level + age + gender + race + admission_type +
  indicator_2_level + num_medications + discharge_disposition + number_diagnoses +
  admission_source + insulin + diabetesMed + A1Cresult + max_glu_serum +
  payer_code + number_outpatient + number_inpatient + number_emergency + metformin 

# Ensure `readmitted` is a factor for classification
TrainData_agg$readmitted <- as.factor(TrainData_agg$readmitted)

# Custom summary function for LogLoss, Accuracy, and Kappa
customSummary <- function(data, lev = NULL, model = NULL) {
  prob_yes <- data$yes
  pred_class <- factor(ifelse(prob_yes > 0.5, "yes", "no"), levels = c("no", "yes"))
  accuracy <- mean(pred_class == data$obs)
  kappa <- confusionMatrix(pred_class, data$obs)$overall["Kappa"]
  logloss <- LogLoss(prob_yes, ifelse(data$obs == "yes", 1, 0))
  out <- c(Accuracy = accuracy, Kappa = kappa, LogLoss = logloss)
  return(out)
}

# Cross-validation settings
trControl <- trainControl(
  method = "cv",
  number = 2,
  classProbs = TRUE,
  summaryFunction = customSummary
)

# Train the MARS model
mars_model <- train(
  formula,
  data = TrainData_agg,
  method = "earth",
  trControl = trControl,
  tuneGrid = expand.grid(degree = 1, nprune = 15),  # Adjust as needed
  glm = list(family = binomial)  # For classification with MARS
)

# Print model results
print(mars_model)

# Predict probabilities on the test data
mars_pred_test <- predict(mars_model, TestData_agg, type = "prob")

# Extract probabilities for the "yes" class (readmission probability)
readmission_probabilities <- mars_pred_test$yes
print(readmission_probabilities)

# Prepare submission
submission <- data.frame(patientID = TestData_agg$patientID, predReadmit = readmission_probabilities)

# Save predictions to CSV
write.csv(submission, "C:/Users/saiab/Desktop/IDA_ASSIGNMENTS/hw_7/MARS_readmission_predict.csv", row.names = FALSE)




