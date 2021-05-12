# Author: Mary Damilola Aiyetigbo
# Date: April 14, 2021
# Purpose: Individual Project (Red Wine Quality)

rm(list = ls())

library(tree)
library(randomForest)
library(ggplot2)
library(gbm)
library(boot)
library(splines)
library(caret)
library(e1071)


##########################Load Dataset to R##################################################
data_url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
download.file(url = data_url, destfile = 'winequality-red.csv')
wineData <- read.csv("winequality-red.csv", sep=";")


#omit any missing data
wineData <- na.omit(wineData)


#attach features
attach(wineData)

################################EXPLORATORY DATA ANALYSIS#############################################

#visualizing the distribution of the response variable "quality"
hist(quality, main = "Quality Distribution", col=c('#0000ff'))

########Visualizing relationship between response variable and some predictors####
QUALITY_ <- factor(quality)
ALCOHOL_ <- factor(round(alcohol))

#Relationship between citric acid, alcohol and quality
ggplot(wineData, aes(x = ALCOHOL_, y = citric.acid, color = QUALITY_)) + geom_boxplot()

#Relationship between volatile acidity, alcohol and quality
ggplot(wineData, aes(x = ALCOHOL_, y = volatile.acidity, color = QUALITY_)) + geom_boxplot()



##########################Splitting Dataset into Train and Test set#########################
set.seed(1)
N <- nrow(wineData)
train <- sample(1:N, N*0.7)
test <- seq(1:N)[-train]


##########################Model 1: Linear Regression#############################################
#GLM Function
#cross validation
set.seed(1)
glm.wine <- glm(quality ~ ., data=wineData, family = "gaussian")
#5-Fold cross validation
cv.error.5 <- cv.glm(wineData, glm.wine, K=5)
cv.error.5$delta[1]

#MSE: 0.4217843



##########################Model 2: Support Vector Machine (SVM)#########################
set.seed(1)
tune.out <- tune(svm, quality ~ ., data=wineData[train,], kernel="radial", 
                 ranges=list(cost=c(0.1, 1, 10, 100, 1000), gamma=c(0.5, 1, 2, 3, 4)))
summary(tune.out)

#best parameters:
#cost gamma
#1   0.5
#svmfit <- svm(quality ~ ., data=wineData[train,], kernel="radial", gamma=0.5, cost=1)

#Perform prediction and calculate MSE
pred.svm=predict(tune.out$best.model, newdata=wineData[test,])
mean((pred.svm-wineData[test,"quality"])^2)
#MSE: 0.4346232


#########################Model 3: Random Forest Tree Model####################################
set.seed(1)
P <- ncol(wineData)-1
rf.wineModel <- randomForest(quality~., data=wineData, subset=train, mtry=sqrt(P), importance=TRUE)
rf.wineModel

#Perform prediction and calculate MSE
pred.rf <- predict(rf.wineModel, newdata = wineData[test,])
mean((pred.rf-wineData[test,"quality"])^2)
#MSE: 0.3603139

#To view the most important predictors used in the model
importance(rf.wineModel)
varImpPlot(rf.wineModel)



#########################USING RANDOM FOREST FOR PREDICTIONS##################
#Prediction using Alcohol
pred.alcohol<-c(8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4)
pred.data <- data.frame(fixed.acidity=mean(fixed.acidity), volatile.acidity=mean(volatile.acidity), citric.acid=mean(citric.acid), 
                        residual.sugar=mean(residual.sugar), chlorides=mean(chlorides), free.sulfur.dioxide=mean(free.sulfur.dioxide), 
                        total.sulfur.dioxide=mean(total.sulfur.dioxide), density=mean(density), pH=mean(pH), 
                        sulphates=mean(sulphates), alcohol=pred.alcohol)

pred.values <- predict(rf.wineModel, pred.data, type="response")
plot(pred.alcohol, pred.values, xlab = "Alcohol", ylab = "Quality of Wine", col = "blue", pch=16)



#Prediction using Sulphate
sulphateValues<-c(0.3, 0.6, 1.2,1.5,1.8,2.0)
pred.sulphate <- data.frame(fixed.acidity=mean(fixed.acidity), volatile.acidity=mean(volatile.acidity), citric.acid=mean(citric.acid), 
                            residual.sugar=mean(residual.sugar), chlorides=mean(chlorides), free.sulfur.dioxide=mean(free.sulfur.dioxide), 
                            total.sulfur.dioxide=mean(total.sulfur.dioxide), density=mean(density), pH=mean(pH), 
                            sulphates=sulphateValues, alcohol=mean(alcohol))

pred.values1 <- predict(rf.wineModel, pred.sulphate, type="response")
plot(sulphateValues, pred.values1, xlab = "Sulphates", ylab = "Quality of Wine", col = "blue", pch=16)



#Prediction using Volatile Acidity
pred.volatile<-c(0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5)
pred.data <- data.frame(fixed.acidity=mean(fixed.acidity), volatile.acidity=pred.volatile, citric.acid=mean(citric.acid), 
                        residual.sugar=mean(residual.sugar), chlorides=mean(chlorides), free.sulfur.dioxide=mean(free.sulfur.dioxide), 
                        total.sulfur.dioxide=mean(total.sulfur.dioxide), density=mean(density), pH=mean(pH), 
                        sulphates=mean(sulphates), alcohol=mean(alcohol))

pred.values <- predict(rf.wineModel, pred.data, type="response")
plot(pred.volatile, pred.values, xlab = "Volatile Acidity", ylab = "Quality of Wine", col = "blue", pch=16)



#Prediction using Chloride
pred.chloride<-c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
pred.data <- data.frame(fixed.acidity=mean(fixed.acidity), volatile.acidity=mean(volatile.acidity), citric.acid=mean(citric.acid), 
                        residual.sugar=mean(residual.sugar), chlorides=pred.chloride, free.sulfur.dioxide=mean(free.sulfur.dioxide), 
                        total.sulfur.dioxide=mean(total.sulfur.dioxide), density=mean(density), pH=mean(pH), 
                        sulphates=mean(sulphates), alcohol=mean(alcohol))

pred.values <- predict(rf.wineModel, pred.data, type="response")
plot(pred.chloride, pred.values, xlab = "Chloride", ylab = "Quality of Wine", col = "blue", pch=16)





