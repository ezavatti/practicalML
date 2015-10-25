---
title: "Practical Machine Learning Project-Data Management"
author: "Elvis Zavatti"
date: "Tuesday, October 13, 2015"
output: html_document
---
## Background

The following project correspond to the project of the course: Practical Machine Learning in the Data Management Specialization at John Hopkins University.

The objective of the project is to use Machine Learning Techniques in order to predict the manner in which a group of people have been exercising, that is, exercising correctly or incorrectly. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

A group of people were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Data were collected using devices such as Jawbone Up, Nike FuelBand, and Fitbit. These devices allow the collection of a large amount of data about personal activity relatively inexpensively. 

They are also part of the quantified self-movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 

The manner in which people did the exercise has been stored in the variable "classe" in the training set. As part of the project, any other variables may be used to predict with.

## Source of data

The data for this project are available in the following link: 

TRAINING DATA: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
TEST DATA:         https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 


## Steps followed to develop the project

1. Load packages to be used
2. Read data base from source
3. Data exploration
4. Cleaning data
5. Load packages to be used
6. Select training dataset and test dataset
7. Apply prediction models
7. Verify the model with the test dataset


```r
library(caret)
library(dplyr)
library(randomForest)
library(e1071)
library(gbm)
```

```r
# Reading from the source and writing database into the working directory

entrena<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
write.csv(entrena, file ="entrena.csv")

# Identifying missing data and setting them to "NA"

entrename<- read.csv("entrena.csv", na.strings = c(na.strings=c("NA","","#DIV/0!")), row.names = 1)

# Reading test data from source. This is for the second part of the project.

prueba<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

write.csv(prueba, file ="prueba.csv")

prueba<- read.csv("prueba.csv", na.strings = c(na.strings=c("NA","","#DIV/0!")), row.names = 1)
```

The training data has 19622 observations and 160 Features and has a large number of NA fields


```r
dim(entrename)
```

```
## [1] 19622   160
```


Now we have the data set for training and test. We will now select the predictors among the features.

The selection of the predictors will be done by eliminating the "near-zero-variance" features.


```r
# exclude near zero variance features, exclude columns with descrptive text, exclude columns with 30% or more of NA.
elimina <- nearZeroVar(entrename)
entrain <- entrename[, -elimina]

longi <- sapply(entrain, function(x) {
    sum(!(is.na(x) | x == ""))
})
fueracol <- names(longi[longi < 0.95*length(entrain$classe)])
coldescr <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
    "cvtd_timestamp", "new_window", "num_window")
eliminado <- c(coldescr, fueracol)
entrain <- entrain[, !names(entrain) %in% eliminado]
```

## Subsetting data Training and cross validation

The data has been subset as follows:
1. Training data is 60% of the full database randomly selected
2. the cross-validation data is 20% of the full database
3. Test data is 20% of the full database

Therefore, model can be tested with two set of "test" data: Cross-validation and test.


```r
set.seed(2000)

entrain$classe<- as.factor(entrain$classe)
inTrain<- createDataPartition(entrain$classe, list = FALSE, p=0.6)
entrain<- entrain[inTrain,]
testy<- entrain[-inTrain,]

inTrain <- createDataPartition(testy$classe, list = FALSE, p = 0.50)
valida <- testy[inTrain,]
testtest <- testy[ -inTrain,]
```

## Applying the model on Training data


```r
modelo <- randomForest(classe ~ ., data = entrain, importance = TRUE, ntrees = 5)
```

Prediction and Confusion matrix for cross-validation data


```r
predice<- predict(modelo, valida)
confusionMatrix(predice, valida$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 671   0   0   0   0
##          B   0 454   0   0   0
##          C   0   0 399   0   0
##          D   0   0   0 395   0
##          E   0   0   0   0 432
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9984, 1)
##     No Information Rate : 0.2854     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000    1.000   1.0000
## Specificity            1.0000   1.0000   1.0000    1.000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000    1.000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000    1.000   1.0000
## Prevalence             0.2854   0.1931   0.1697    0.168   0.1838
## Detection Rate         0.2854   0.1931   0.1697    0.168   0.1838
## Detection Prevalence   0.2854   0.1931   0.1697    0.168   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000    1.000   1.0000
```

Prediction and Confusion matrix for test data


```r
predice<- predict(modelo, testtest)
confusionMatrix(predice, testtest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 671   0   0   0   0
##          B   0 453   0   0   0
##          C   0   0 399   0   0
##          D   0   0   0 394   0
##          E   0   0   0   0 432
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9984, 1)
##     No Information Rate : 0.2857     
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
## Prevalence             0.2857   0.1928   0.1699   0.1677   0.1839
## Detection Rate         0.2857   0.1928   0.1699   0.1677   0.1839
## Detection Prevalence   0.2857   0.1928   0.1699   0.1677   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```


## Conclusion

Random forest prediction shows the maximum accuracy and therefore this is the model to be used. 
Model has been trained with the training data and tested with cross-validation data and test data (out-of-sample)


## Assigment: Predictions using the test file from source


```r
pml_write_files = function(x){
n = length(x)
for(i in 1:n){
filename = paste0("problem_id_",i,".txt")
write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
}
x <- prueba

answers <- predict(modelo, newdata=x)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files(answers)
```


## References

[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Article is available at: 

http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf

or

http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201

