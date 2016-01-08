library(jsonlite)
library(tm)
library(gbm)
library(xgboost)
library(SnowballC)
library(Matrix)
library(caret)
library(e1071)
library(foreach)
library(doParallel)
library(randomForest)

set.seed(123456)

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
c_ingredientsDTM <- DocumentTermMatrix(c_ingredients2)
c_ingredientsDTM <- removeSparseTerms(c_ingredientsDTM, 1-3/nrow(c_ingredientsDTM))
c_ingredientsDTM <- as.data.frame(as.matrix(c_ingredientsDTM))
#######

#######feature engineering (basic)
c_ingredientsDTM$ingredients_count  <- rowSums(c_ingredientsDTM) 
c_ingredientsDTM$cuisine <- as.factor(dat$cuisine)
c_ingredientsDTM_Model <- c_ingredientsDTM[1:nrow(train),]
c_ingredientsDTM_Submit <- c_ingredientsDTM[-(1:nrow(train)),]
#run this to create sample for testing
#selected<-sample(nrow(c_ingredientsDTM_Model),25000) 
#c_ingredientsDTM_Model <- c_ingredientsDTM_Model[selected,]
#c_ingredientsDTM_Test <- (c_ingredientsDTM[1:nrow(train),])[-selected,]
#######

#######creating ranges to test for random hyperparameter search
#x <- c("max.depth",3,7,0,"eta",.05,.2,3,"subsample",.7,.99,3,"colsample_bytree",.1,.5,3)
#x <- c("max.depth",3,7,0,"eta",.02,.07,3,"subsample",.7,.99,3,"colsample_bytree",.1,.5,3)
#x <- c("max.depth",6,9,0,"eta",.02,.04,3,"subsample",.85,.99,3,"colsample_bytree",.1,.5,3)
x <- c("max.depth",6,8,0,"eta",.025,.035,3,"subsample",.95,.999,3,"colsample_bytree",.05,.3,3)
#######

#######function that creates a list of hyperparameters to test
randParams <- function(x,n){
  y <- data.frame(matrix(x,nrow=4),stringsAsFactors=F)
  names(y) <- y[1,]
  y <- y[-1,]
  y <- data.frame(sapply(y,FUN=as.numeric))
  z <- data.frame(sapply(y,FUN=function(f) {round(runif(n,min=f[1],max=f[2]),f[3])}))
  return(apply(z,1,list))
}
#######

#######setting up a cross validation data set
samp <- sample(c(1,2,3),nrow(c_ingredientsDTM_Model),replace=T)
xs <- Matrix(data.matrix(c_ingredientsDTM_Model[, !colnames(c_ingredientsDTM_Model) %in% c("cuisine")]),sparse = T)

xs_train <- list()
xs_pred <- list()
for(j in 1:3){
  xs_train[j] <- Matrix(data.matrix(c_ingredientsDTM_Model[samp!=j, !colnames(c_ingredientsDTM_Model) %in% c("cuisine")]),sparse = T)
  xs_pred[j] <- Matrix(data.matrix(c_ingredientsDTM_Model[samp==j, !colnames(c_ingredientsDTM_Model) %in% c("cuisine")]),sparse = T)
}
ys <- as.numeric(c_ingredientsDTM_Model$cuisine)-1
#######
# c_ingredientsDTM <- NULL
# c_ingredientsDTM_Model <- NULL
# c_ingredientsDTM_Submit <- NULL
# dat <- NULL
# train <- NULL
# test <- NULL
# c_ingredients <- NULL
# gc()
# 
# cl <- makeCluster(7)-
# registerDoParallel(cl)

#######loop that tests different hyperparameters. outputs results to text file as it loops for monitoring. WARNING, this takes a couple of days to run
t1 <- Sys.time()
xgb_results <- foreach(i=randParams(x,100),.combine = rbind) %do% {
  totCorrect <- 0
  nr <- round(runif(1,3000,5000),0)
  for(j in 1:3){
    xgbmat <- xgb.DMatrix(xs_train[[j]], label=ys[samp!=j])
    xgb <- xgboost(xgbmat, nround = nr, objective = "multi:softmax", num_class = 20,verbose = 0, params = i[[1]])
    pred <- predict(xgb, newdata = xs_pred[[j]])
    totCorrect <- totCorrect + sum(pred==ys[samp==j])
  }
  line <- c(i[[1]],nround = nr,score=totCorrect/length(ys),elapsed_time=(Sys.time()-t1))
  print(line)
  write(line,file="xgb_results.csv",append=TRUE, sep=",",ncolumns=length(line))
  flush.console()
  line
}
t2 <- Sys.time()
t2-t1
#######

# stopCluster(cl)

#######Run Model (run on full data for submission)
xgbmat <- xgb.DMatrix(xs, label=ys)
xgb <- xgboost(xgbmat, nround = 3870, objective = "multi:softmax", num_class = 20,verbose = 1, max.depth = 7, eta = .029, subsample=.99,colsample_bytree=.154)
#######

#######Prediction Testing
pred <- predict(xgb, newdata = data.matrix(c_ingredientsDTM_Test[, !colnames(c_ingredientsDTM_Test) %in% c("cuisine")]))
pred_class <- levels(c_ingredientsDTM_Model$cuisine)[pred+1]
pred_class <- as.factor(pred_class)
summary(pred_class)
sum(as.character(pred_class)==as.character(c_ingredientsDTM_Test$cuisine))/nrow(c_ingredientsDTM_Test)

table(as.character(pred_class),as.character(c_ingredientsDTM_Test$cuisine))
train[rownames(c_ingredientsDTM_Test[pred_class=="italian" & c_ingredientsDTM_Test$cuisine=="japanese",]),]
#######

#######Create predictions for Submission
pred <- predict(xgb, newdata = data.matrix(c_ingredientsDTM_Submit[, !colnames(c_ingredientsDTM_Submit) %in% c("cuisine")]))
pred_class <- levels(c_ingredientsDTM_Model$cuisine)[pred+1]
pred_class <- as.factor(pred_class)
#######

#######Create Submission file
sub <- data.frame(id=test$id,cuisine=pred_class)
summary(sub)

write.csv(sub,"sub",row.names=F)
