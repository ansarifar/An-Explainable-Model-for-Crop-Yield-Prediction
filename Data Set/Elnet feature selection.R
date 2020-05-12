
library(glmnet)  # Package to fit ridge/lasso/elastic net models
library(data.table)
library(ranger)
library(caret)
library(Metrics)

Pre_year=2018;
#seed='Corn';
seed='Soybeans';

if (seed=='Corn') {
  INFO_GM=fread(paste("Corn_INFO_GM",as.character(Pre_year),".csv",sep=""),header=TRUE)
  INFO_SOIL=fread(paste("Corn_INFO_SOIL",as.character(Pre_year),".csv",sep=""),header=TRUE)
  INFO_Weather=fread(paste("Corn_INFO_Weather",as.character(Pre_year),".csv",sep=""),header=TRUE)
  INFO_Yield=fread(paste("Corn_INFO_Yield",as.character(Pre_year),".csv",sep=""),header=TRUE)
  Start=1990
} else {
  INFO_GM=fread(paste("Soybeans_INFO_GM",as.character(Pre_year),".csv",sep=""),header=TRUE)
  INFO_SOIL=fread(paste("Soybeans_INFO_SOIL",as.character(Pre_year),".csv",sep=""),header=TRUE)
  INFO_Weather=fread(paste("Soybeans_INFO_Weather",as.character(Pre_year),".csv",sep=""),header=TRUE)
  INFO_Yield=fread(paste("Soybeans_INFO_Yield",as.character(Pre_year),".csv",sep=""),header=TRUE)
  Start=1995
}


Index_Train=INFO_Yield[,1]>=Start  & INFO_Yield[,1]<=(Pre_year-3)
Index_Valiation=INFO_Yield[,1]>=(Pre_year-2)  & INFO_Yield[,1]<=(Pre_year-1)
Index_Test=INFO_Yield[,1]==Pre_year

XX=cbind(INFO_SOIL,INFO_Weather[,85:280],INFO_GM[,c(1:2,15:28)])

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

XX <- as.data.frame(lapply(XX, normalize))
FF=XX
XX[,which(is.na(FF[1,]))]=NULL


X_Train=XX[Index_Train,]
X_Validation=XX[Index_Valiation,]
X_Test=XX[Index_Test,]
Y_Train=data.frame(Y=INFO_Yield$Yield[Index_Train])
Y_Validation=data.frame(Y=INFO_Yield$Yield[Index_Valiation])
Y_Test=data.frame(Y=INFO_Yield$Yield[Index_Test])

#####################################################################################

X_Trainp=rbind(X_Train,X_Validation)
Y_Trainp=rbind(Y_Train,Y_Validation)

set.seed(125) 
cv.lasso <- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 1, family = "gaussian")
cv.ridge <- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0, family = "gaussian")
cv.elnet1<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.1, family = "gaussian")
cv.elnet2<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.2, family = "gaussian")
cv.elnet3<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.3, family = "gaussian")
cv.elnet4<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.4, family = "gaussian")
cv.elnet5<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.5, family = "gaussian")
cv.elnet6<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.6, family = "gaussian")
cv.elnet7<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.7, family = "gaussian")
cv.elnet8<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.8, family = "gaussian")
cv.elnet9<- cv.glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), alpha = 0.9, family = "gaussian")


fit.lasso <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=1,lambda = cv.lasso$lambda.min)
fit.ridge <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=0,lambda = cv.ridge$lambda.min)
fit.elnet1 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.1,lambda = cv.elnet1$lambda.min)
fit.elnet2 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.2,lambda = cv.elnet2$lambda.min)
fit.elnet3 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.3,lambda = cv.elnet3$lambda.min)
fit.elnet4 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.4,lambda = cv.elnet4$lambda.min)
fit.elnet5 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.5,lambda = cv.elnet5$lambda.min)
fit.elnet6 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.6,lambda = cv.elnet6$lambda.min)
fit.elnet7 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.7,lambda = cv.elnet7$lambda.min)
fit.elnet8 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.8,lambda = cv.elnet8$lambda.min)
fit.elnet9 <- glmnet(as.matrix(X_Trainp), as.matrix(Y_Trainp), family="gaussian", alpha=.9,lambda = cv.elnet9$lambda.min)


yhat0 <- predict(fit.lasso, newx=as.matrix(X_Test))
yhat1 <- predict(fit.ridge, newx=as.matrix(X_Test))
yhat2 <- predict(fit.elnet1, newx=as.matrix(X_Test))
yhat3 <- predict(fit.elnet2, newx=as.matrix(X_Test))
yhat4 <- predict(fit.elnet3, newx=as.matrix(X_Test))
yhat5 <- predict(fit.elnet4, newx=as.matrix(X_Test))
yhat6 <- predict(fit.elnet5, newx=as.matrix(X_Test))
yhat7 <- predict(fit.elnet6, newx=as.matrix(X_Test))
yhat8 <- predict(fit.elnet7, newx=as.matrix(X_Test))
yhat9 <- predict(fit.elnet8, newx=as.matrix(X_Test))
yhat10 <- predict(fit.elnet9, newx=as.matrix(X_Test))

mse0 <- sqrt(mean((Y_Test$Y - yhat0)^2))
mse1 <- sqrt(mean((Y_Test$Y - yhat1)^2))
mse2 <- sqrt(mean((Y_Test$Y - yhat2)^2))
mse3 <- sqrt(mean((Y_Test$Y - yhat3)^2))
mse4 <- sqrt(mean((Y_Test$Y - yhat4)^2))
mse5 <- sqrt(mean((Y_Test$Y - yhat5)^2))
mse6 <- sqrt(mean((Y_Test$Y - yhat6)^2))
mse7 <- sqrt(mean((Y_Test$Y - yhat7)^2))
mse8 <- sqrt(mean((Y_Test$Y - yhat8)^2))
mse9 <- sqrt(mean((Y_Test$Y - yhat9)^2))
mse10 <- sqrt(mean((Y_Test$Y - yhat10)^2))

mse0
mse1
min(c(mse2,mse3,mse4,mse5,mse6,mse7,mse8,mse9))


#########################################################################################################


XX_all_train=cbind(X_Trainp,Y_Trainp)
linearMod <- lm(Y ~ ., XX_all_train)
yhat0p <- predict(linearMod, cbind(X_Test,Y_Test))
mse0p <- sqrt(mean((Y_Test$Y - yhat0p)^2))
mse0p


