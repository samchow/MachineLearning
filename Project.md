# Machine Learning Project
Cheeseng Chow  
February 21, 2015  

## Synopsis
We use the random forest machine learning algorithm to predict the kinds of exercise ("classe") from smartphone data. We will also measure the efficacy of the algorithm.

## Overview
We will first load caret and random forest and read in the training and test data.

```r
library(caret)
library(randomForest)
pml.train <- read.csv("pml-training.csv")
```

## Cross Validation
For cross validation, we'll randomly select 25% of the train data for testing and use the remaining 75% for model training. We will use the test data to predict how well the algorithm perform on the real test data.

```r
set.seed(12345)
inTrain <- createDataPartition(y=pml.train$classe, p=0.75, list=FALSE)
train <- pml.train[inTrain,]
test <- pml.train[-inTrain,]
```

## Relevant Columns 
Instead of removing columns from the training data, we will identify relevant columns and construct an appropriate formula using only the relevant formula. The basic criteria is the column should not have any NA or blanks:

```r
colnames(pml.train[,!sapply(pml.train, function(x) any(is.na(x) | x ==""))])
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

We will ignore the first 7 columns as well. So the training formula is as follows:

```r
formula <- classe ~ roll_belt + pitch_belt + yaw_belt + total_accel_belt + gyros_belt_x +
  gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y +
  accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z +
  roll_arm + pitch_arm + yaw_arm + total_accel_arm +
  gyros_arm_x + gyros_arm_y + gyros_arm_z + accel_arm_x +
  accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y +
  magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell +
  total_accel_dumbbell + gyros_dumbbell_x + gyros_dumbbell_y + gyros_dumbbell_z +
  accel_dumbbell_x + accel_dumbbell_y + accel_dumbbell_z + magnet_dumbbell_x +
  magnet_dumbbell_y + magnet_dumbbell_z + roll_forearm + pitch_forearm +
  yaw_forearm + total_accel_forearm + gyros_forearm_x + gyros_forearm_y +
  gyros_forearm_z + accel_forearm_x + accel_forearm_y + accel_forearm_z +
  magnet_forearm_x + magnet_forearm_y + magnet_forearm_z  
```

## Training
Now apply random forest algorithm to the training data

```r
modelFit <- randomForest(formula, data=train)
```

## In-Sample Accuracy
We apply the resultant model on the training set and run the confusionMatrix() on the prediction.
We note that the random forest model has 100% accuracy. 

```r
pTrain <- predict(modelFit, newdata=train)
confusionMatrix(pTrain, train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```


## Cross Validation and Out-of-Sample Errors
We now apply the prediction algorithm to our test sets.

```r
pTest <- predict(modelFit,newdata=test)
confusionMatrix(pTest, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    6    0    0    0
##          B    1  938    2    0    0
##          C    0    5  851    7    1
##          D    0    0    2  797    4
##          E    0    0    0    0  896
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9943          
##                  95% CI : (0.9918, 0.9962)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9928          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9884   0.9953   0.9913   0.9945
## Specificity            0.9983   0.9992   0.9968   0.9985   1.0000
## Pos Pred Value         0.9957   0.9968   0.9850   0.9925   1.0000
## Neg Pred Value         0.9997   0.9972   0.9990   0.9983   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1913   0.1735   0.1625   0.1827
## Detection Prevalence   0.2855   0.1919   0.1762   0.1637   0.1827
## Balanced Accuracy      0.9988   0.9938   0.9961   0.9949   0.9972
```

We note that it has an overall accuracy of 99.43%. That is the accuracy we expect on the actual test data.


# Appendix -- Actual Test Predictions
We now generate the actual test predictions and write it out for submission

```r
pml.test <- read.csv("pml-testing.csv")
answers <- predict(modelFit, newdata=pml.test)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```

The 20 submitted answers were all correct. This is expected since our cross-validation data suggest that  model is 99+% accurate.
