#####
#Kaggle

library(jsonlite)
library(dplyr)
library(ggplot2)
library(tm) # For NLP; creating bag-of-words
library(caret)
library(rpart)
library(rpart.plot)
library(SnowballC)

setwd("C:\\Users\\ERR763\\Desktop\\Kaggle\\Cooking\\train")
train <- fromJSON("train.json", flatten = TRUE)

ggplot(data = train, aes(x = cuisine)) +   geom_histogram() + labs(title = "Cuisines", x = "Cuisine", y = "Number of Recipes")

ingredients <- )
Corpus(VectorSource(train$ingredients)
       ingredients <- tm_map(ingredients, stemDocument)
       
       ingredientsDTM <- DocumentTermMatrix(ingredients)
       
       #Feature selection
       sparse <- removeSparseTerms(ingredientsDTM, 0.99)
       ## This function takes a second parameters, the sparsity threshold.
       ## The sparsity threshold works as follows.
       ## If we say 0.98, this means to only keep terms that appear in 2% or more of the recipes.
       ## If we say 0.99, that means to only keep terms that appear in 1% or more of the recipes.
       
       ingredientsDTM <- as.data.frame(as.matrix(sparse))
       ## Add the dependent variable to the data.frame
       ingredientsDTM$cuisine <- as.factor(train$cuisine)
       
       inTrain <- createDataPartition(y = ingredientsDTM$cuisine, p = 0.6, list = FALSE)
       training <- ingredientsDTM[inTrain,]
       testing <- ingredientsDTM[-inTrain,]
       
       