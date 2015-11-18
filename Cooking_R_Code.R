library(jsonlite)
library(tm)
library(gbm)
library(xgboost)
library(SnowballC)
library(Matrix)
library(caret)
library(e1071)

set.seed(1234)

######pull in data
setwd("~/Kaggle/Text_Chefs/Cooking")
train <- fromJSON("train/train.json", flatten = TRUE)
test <- fromJSON("test/test.json", flatten = TRUE)
test$cuisine <- "italian"
dat <- rbind(train,test)
#######

#######scrub text
dat$ingredients <- lapply(dat$ingredients, FUN=tolower)
dat$ingredients <- lapply(dat$ingredients, FUN=function(x) gsub("-", "_", x)) # allow dash e.g. "low-fat"
dat$ingredients <- lapply(dat$ingredients, FUN=function(x) gsub("[^a-z0-9_ ]", "", x)) # allow regular character and spaces
#this creates multi-word variables
dat$ingredients <- lapply(dat$ingredients, FUN=function(x) paste(x,gsub(" ", "_", x)))
c_ingredients <- Corpus(VectorSource(dat$ingredients))
#######

#######create simple DTM
c_ingredientsDTM <- DocumentTermMatrix(c_ingredients)
c_ingredientsDTM <- removeSparseTerms(c_ingredientsDTM, 1-3/nrow(c_ingredientsDTM))
c_ingredientsDTM <- as.data.frame(as.matrix(c_ingredientsDTM))
#need to take sign to not double count single and multi-word
c_ingredientsDTM <- sign(c_ingredientsDTM)
#######

#######feature engineering (basic)
c_ingredientsDTM$ingredients_count  <- rowSums(c_ingredientsDTM) 
c_ingredientsDTM$cuisine <- as.factor(dat$cuisine)
c_ingredientsDTM_Model <- c_ingredientsDTM[1:nrow(train),]
c_ingredientsDTM_Submit <- c_ingredientsDTM[-(1:nrow(train)),]
#run this to create sample for testing
#selected<-sample(nrow(c_ingredientsDTM_Model),25000) 
#c_ingredientsDTM_Model <- c_ingredientsDTM_Model[selected,]
#c_ingredientsDTM_Test <- c_ingredientsDTM_Model[-selected,]
#######

#matrix for models
xgbmat <- xgb.DMatrix(Matrix(data.matrix(c_ingredientsDTM_Model[, !colnames(c_ingredientsDTM_Model) %in% c("cuisine")])), label=as.numeric(c_ingredientsDTM_Model$cuisine)-1)


#######Run these for model tunning.  Don't need to run each time.
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        #summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)

xgb.grid <- expand.grid(nrounds = 1000,
                        eta = c(0.05,0.1,0.2,0.3),
                        max_depth = c(10,15,20,25,30)
)
xgb_tune <-train(cuisine~.,
                 data=c_ingredientsDTM_Model,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="Kappa",
                 nthread =7
              )
xgb_tune$bestTune
#max_depth 10 eta .05

xgb.grid <- expand.grid(nrounds = 1000,
                        eta = c(0.01,0.03,0.05,0.07),
                        max_depth = c(4,6,8,10,12)
)
xgb_tune <-train(cuisine~.,
                 data=c_ingredientsDTM_Model,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="Kappa",
                 nthread =7
)
xgb_tune$bestTune
#max_depth 4 eta .07

xgb.grid <- expand.grid(nrounds = 1500,
                        eta = c(0.06,0.07,0.08,0.09),
                        max_depth = c(2,3,4,5)
)
xgb_tune <-train(cuisine~.,
                 data=c_ingredientsDTM_Model,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="Kappa",
                 nthread =7
)
xgb_tune$bestTune
#max_depth 5 eta .09

xgb.grid <- expand.grid(nrounds = c(500,750,1000,1500,2000,2500),
                        eta = c(0.09),
                        max_depth = c(5)
)
xgb_tune <-train(cuisine~.,
                 data=c_ingredientsDTM_Model,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="Kappa",
                 nthread =1
)
xgb_tune$bestTune
#nrounds 1500.  Submission on full data got better with 2000.
#######

#######Run Model (run on full data for submission)
xgb <- xgboost(xgbmat, max.depth = 5, eta = 0.09, nround = 2000, objective = "multi:softmax", num_class = 20)
#######

#######Prediction Testing
pred <- predict(xgb, newdata = data.matrix(c_ingredientsDTM_Test[, !colnames(c_ingredientsDTM_Test) %in% c("cuisine")]))
pred_class <- levels(c_ingredientsDTM_Model$cuisine)[pred+1]
pred_class <- as.factor(pred_class)
summary(pred_class)
sum(as.character(pred_class)==as.character(c_ingredientsDTM_Test$cuisine))/nrow(c_ingredientsDTM_Test)

table(as.character(pred_class),as.character(c_ingredientsDTM_Test$cuisine))
c_ingredientsDTM_Test[pred_class=="vietnamese" & c_ingredientsDTM_Test$cuisine=="thai",]
train[1454,]
#######

#######Create predictions for Submission
pred <- predict(xgb, newdata = data.matrix(c_ingredientsDTM_Submit[, !colnames(c_ingredientsDTM_Submit) %in% c("cuisine")]))
pred_class <- levels(c_ingredientsDTM_Model$cuisine)[pred+1]
pred_class <- as.factor(pred_class)
#######

#######Create Submission file
sub <- data.frame(id=test$id,cuisine=pred_class)
summary(sub)

write.csv(sub,"sub5",row.names=F)
