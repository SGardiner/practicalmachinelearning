# Machine Learning - Project 1
Sam Gardiner  
July 22, 2016  



## Dataset Description
The training dataset has 19622 rows and 160 columns, while the testing dataset (finalTest) is 20 rows, 160 columns.  The training dataset outcome variable "classe" is a factor with 5 different levels: A,B,C,D and E.  A is the classification for the exercise performed correctly, while B through E are different ways the exercise is performed incorrectly. Since the outcomes are not continuous, accuracy measurements (sensitivity and specificity) will be relevant and error measurements like RMSE (Mean Square Error) will not be relevant. 

```r
# Read training / testing datasets from http://groupware.les.inf.puc-rio.br/har
training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
finalTest <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))
```

## Cross validation
A cross validation strategy would be to develop prediction models based on a split of the training set into a training (training2) and test dataset (testing2), using the createDataPartition function.  The split chosen was 70% training (13737 rows) and 30% testing (5885).

```r
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
# take original training set and split into training (70%) and testing (30%)
training2 <- training[inTrain,]
testing2 <- training[-inTrain,]
```

## Preparing the dataset
The original downloaded dataset (finalTest) is only 20 rows.  A cursory look at the data shows that majority of the columns (100 out of 160) are completely filled with NA entries.  Since these columns won't be important for the final testing, they're not relevant for the fitted model and can be eliminated for the training2 dataset.  In addition, the columns 1,2,5,6 and 7 are extraneous variables (user_name, timestamps and windows) and can be eliminated from the training2 dataset.


```r
NAcol <- NULL
for (i in 1:ncol(training)) {
        if(sum(!is.na(finalTest[,i])) == 0) {NAcol <- c(NAcol, i)}
}
# Columns 1,2,5,6,7 are also eliminated from the training sets
NAcol <- c(1,2,5:7,NAcol)
training2 <- training2[,-NAcol]
```

To speed up evaluation of different machine learning algorithms, a smaller training dataset (smallTrain) was created that is a 5% sampling of the training2 dataset, with 689 rows and 55 columns.

```r
# create small practice training set
inTrain2 <- createDataPartition(y = training2$classe, p = 0.05, list = FALSE)
smallTrain <- training2[inTrain2,]
```

## Building and evaluating prediction models
Using the smallTrain dataset, the following machine learning algorithms were evaluated for processing speed and accuracy:  
- recursive partitioning (rpart)  
- boosted trees (gbm)  
- linear discriminant analysis (lda)  
- random forests (rf)  
- bagging (treebag)  
  
While model training was done on the smallTrain dataset, evaluation of accuracy was performed on the full testing2 dataset (5885 observations, 160 variables)

Note: for performance reasons on the random forests machine learning algorithm, the randomForest package was used instead of the caret package train(method = "rf") function.


```r
# rpartModFit - recursive partitioning method
ptm <- proc.time()
rpartModFit <- train(classe ~., data = smallTrain, method = "rpart")
rpartPtm <- (proc.time() - ptm)[3]  # record total elapsed time

# gbmModFit - boosted trees method
ptm <- proc.time()
gbmModFit <- train(classe ~., data = smallTrain, method = "gbm", verbose = FALSE)
gbmPtm <- (proc.time() - ptm)[3]

# ldaModFit - linear discriminant analysis method
ptm <- proc.time()
ldaModFit <- train(classe ~., data = smallTrain, method = "lda")
ldaPtm <- (proc.time() - ptm)[3]

# rfModFit - random forests, k-fold (number = 5) cross validation
ptm <- proc.time()
rfModFit <- train(classe ~., data = smallTrain, method = "rf", trControl = trainControl(method = "cv", number = 5), ntree = 100)
rfPtm <- (proc.time() - ptm)[3]

# bagModFit - bagging method using "treebag" method
ptm <- proc.time()
bagModFit <- train(classe ~., data = smallTrain, method = "treebag")
bagPtm <- (proc.time() - ptm)[3]
```



From the results of machine learning algorithms, the "rpart" and "lda" methods had fast processing times, but were only 50% - 70% accurate.  The "gbm", "rf" and "bag" methods all had about 92% accuracy, but the "rf" method had the fastest processing time of the three (even using the "cv" cross validation method), helped by optimization of some of the parameters. The "gbm" method had by far the longest processing time.


```
##       elapsed accuracy
## rpart   1.816   0.4693
## gbm    66.056   0.9193
## lda     1.036   0.6653
## rf      4.253   0.9186
## bag    19.529   0.9264
```

![](index_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

## In sample and out of sample accuracy (error)
For the final evaluation, the randomForest algorithm is used on the full training2 data set.  The processing time on the full 13737 row training2 dataset was approximately 120 seconds on my computer.  Cross validation was used (method = "cv", number = 5), which added to the processing time.  The in sample model accuracy is very high though at 0.99687.  The error rate is (1 - accuracy) = 0.00313.  

```r
ptm <- proc.time()
# rfModFit2 <- randomForest(classe ~., data = training2)
rfModFit2 <- train(classe ~., data = training2, method = "rf", trControl = trainControl(method = "cv", number = 5), ntree = 100)
rfPtm2 <- (proc.time() - ptm)[3]
print(paste("Processing time for rf method on training2 dataset:",rfPtm2))
```

```
## [1] "Processing time for rf method on training2 dataset: 119.602"
```

```r
print(paste("In sample accuracy of the random forests algorithm:", round((rfModFit2$results[2,2]),5)))
```

```
## [1] "In sample accuracy of the random forests algorithm: 0.99687"
```

The estimated out of sample error rate should be very close to the in sample error, because cross validation was used in fitting the model.  The results below show this to be the case with an error rate of 0.00187, or an accuracy of 0.99813.

```r
rfPred2 <- predict(rfModFit2, testing2)
rfAccuracy2 <- round(sum(rfPred2 == testing2$classe)/nrow(testing2),5)
print(paste("Out of sample accuracy of random forests algorithm:", rfAccuracy2))
```

```
## [1] "Out of sample accuracy of random forests algorithm: 0.99813"
```

```r
print(confusionMatrix(testing2$classe, rfPred2)) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    3 1136    0    0    0
##          C    0    3 1022    1    0
##          D    0    0    2  960    2
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9981          
##                  95% CI : (0.9967, 0.9991)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9976          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9974   0.9980   0.9990   0.9982
## Specificity            1.0000   0.9994   0.9992   0.9992   1.0000
## Pos Pred Value         1.0000   0.9974   0.9961   0.9959   1.0000
## Neg Pred Value         0.9993   0.9994   0.9996   0.9998   0.9996
## Prevalence             0.2850   0.1935   0.1740   0.1633   0.1842
## Detection Rate         0.2845   0.1930   0.1737   0.1631   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9991   0.9984   0.9986   0.9991   0.9991
```

## Further optimization of the predictors used in the random forest model
To speed processing time, other predictors (variables) can sometimes be eliminated due to lack of variability.  Using the nearZeroVar function, the variables were tested for zero variance or near zero variance.  All resulted in FALSE, meaning none could be eliminated for this reason.

```r
nzv <- nearZeroVar(training2, saveMetrics = TRUE)
print(paste(sum(nzv$nzv + nzv$zeroVar),"predictors are removed due to lack of variability."))
```

```
## [1] "0 predictors are removed due to lack of variability."
```

Next, the varImp function was used to identify the most important variables for the random forest prediction model to see whether the rest can be eliminated.

```r
print(varImp(rfModFit2))
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 54)
## 
##                      Overall
## raw_timestamp_part_1 100.000
## roll_belt             42.323
## pitch_forearm         26.279
## yaw_belt              18.676
## magnet_dumbbell_z     18.524
## magnet_dumbbell_y     18.214
## pitch_belt            16.031
## roll_forearm          15.367
## accel_dumbbell_y       8.517
## accel_forearm_x        6.616
## roll_dumbbell          6.384
## accel_belt_z           6.049
## magnet_dumbbell_x      5.946
## magnet_belt_y          5.661
## magnet_belt_z          5.375
## total_accel_dumbbell   5.357
## accel_dumbbell_z       5.117
## magnet_forearm_z       4.075
## gyros_belt_z           3.975
## yaw_dumbbell           3.271
```

From the results, it appears the top 8 variables have the greatest effect on the model.  The subsequent model built with just these 8 variables have essentially the same accuracy (99.68%) as the full training2 dataset with 54 variables, but runs in less than 1/6 of the the time at 18 seconds.  This was a really impressive result in reduction of processing time, with no tradeoff in error rate.

```r
# columns numbers in order of greatest importance
import = order(varImp(rfModFit2)$importance, decreasing = TRUE)[1:8]

# traing the model on the training2 dataset with only the top 8 variables
ptm <- proc.time()
rfModFit3 <- train(classe ~., data = training2[,c(import,55)], method = "rf", trControl = trainControl(method = "cv", number = 5), ntree = 100)
rfPtm3 <- (proc.time() - ptm)[3]
print(paste("Processing time for rf method on training2 dataset:",rfPtm3))
```

```
## [1] "Processing time for rf method on training2 dataset: 18.358"
```

```r
print(paste("In sample accuracy of the random forests algorithm:", round((rfModFit3$results[2,2]),5)))
```

```
## [1] "In sample accuracy of the random forests algorithm: 0.9968"
```

## Final results on 20 test cases
Final results of the rfModFit2 model are run on the original testing dataset (finalTest), with 20 rows of data.  Results are 20 / 20, 100% accurate.

```r
rfPredFinal <- predict(rfModFit2, finalTest[,-c(160)])

finalResults <- as.data.frame(matrix(nrow = 1, ncol = 20))
rownames(finalResults) <- c("results")
colnames(finalResults) <- finalTest$problem_id
finalResults <- rfPredFinal

print("Accuracy on original testing dataset is 20 / 20"); print(finalResults)
```

```
## [1] "Accuracy on original testing dataset is 20 / 20"
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


