library(caret)
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainDest <- "./groupware/train.csv"
testDest <- "./groupware/test.csv"

if(!exists(trainDest)){
        download.file(url = trainURL, destfile = trainDest)
}

if(!exists(testDest)){
        download.file(url = testURL, destfile = testDest)
}

trainRaw <- read.csv(trainDest,na.string=c("#DIV/0!","NA"))
testRaw <- read.csv(testDest,na.string=c("#DIV/0!","NA"))

names(trainRaw[1:7])

names(t)

## why not displaying?

trainRaw2 <-trainRaw[-c(1:7)]
testRaw2 <-testRaw[-c(1:7)]

noNaColumns <-which(colSums(is.na(trainRaw2[,-153]))==0)
training <- cbind(trainRaw2[,noNaColumns],classe = trainRaw$classe)
testing <- testRaw2[,noNaColumns]

set.seed(456)
dim(training)
str(training)



inTrain <- createDataPartition(training$classe, p = .70, list = FALSE)
trainPart <- training[inTrain,]
testPart <- training[-inTrain,]

prePro <- preProcess(trainPart, method = "pca" ,thresh = 0.8)
trainPartPC <- predict(prePro, trainPart)
testPartPC <- predict(prePro, testPart)


fitControl <- trainControl(## 10-fold CV
        method = "none",
        number = 10,
        preProcOptions =list(thresh = .8),
        ## repeated ten times
        repeats = 10)


amdai <- train(classe ~., data = trainPartPC, method = "amdai")#good
amdai
treebag <- train(classe ~., data = trainPart, method = "treebag")#really good
treebag
predict(treebag,testing)
bayesglm <- train(classe ~., data = trainPartPC, method = "bayesglm")#good
bayesglm
BstLm <- train(classe ~., data = trainPartPC, method = "BstLm")#good
BstLm
bstTree <- train(classe ~., data = trainPartPC, method = "bstTree")#good
bstTree
J48 <- train(classe ~., data = trainPartPC, method = "J48")#good
J48
rpart <- train(classe ~., data = trainPartPC, method = "rpart")#good
rpart
rpart1SE <- train(classe ~., data = trainPartPC, method = "rpart1SE")#good
rpart1SE
rpart2 <- train(classe ~., data = trainPartPC, method = "rpart2")#good
rpart2
rpartScore <- train(classe ~., data = trainPartPC, method = "rpartScore")#good
rpartScore
ctree <- train(classe ~., data = trainPartPC, method = "ctree")#good
ctree
ctree2 <- train(classe ~., data = trainPartPC, method = "ctree2")#good
ctree2
vglmContRatio <- train(classe ~., data = trainPartPC, method = "vglmContRatio")#good
vglmContRatio
xgbTree <- train(classe ~., data = trainPartPC, method = "xgbTree")#good
xgbTree
elm <- train(classe ~., data = trainPartPC, method = "elm")#good
elm
fda <- train(classe ~., data = trainPartPC, method = "fda")#good
fda
FH.GBML <- train(classe ~., data = trainPartPC, method = "FH.GBML")#testing
hda <- train(classe ~., data = trainPartPC, method = "hda")#good
hda
hdda <- train(classe ~., data = trainPartPC, method = "hdda")#good
hdda
kknn <- train(classe ~., data = trainPartPC, method = "kknn")#really good
kknn
knn <- train(classe ~., data = trainPartPC, method = "knn")#good
knn
lvq <- train(classe ~., data = trainPartPC, method = "lvq")#good
lvq
lda <- train(classe ~., data = trainPartPC, method = "lda")#good
lda
lda2 <- train(classe ~., data = trainPartPC, method = "lda2")#good
lda2
stepLDA <- train(classe ~., data = trainPartPC, method = "stepLDA")#good
stepLDA
LMT <- train(classe ~., data = trainPartPC, method = "LMT")
Mlda <- train(classe ~., data = trainPartPC, method = "Mlda")
mda <- train(classe ~., data = trainPartPC, method = "mda")
manb <- train(classe ~., data = trainPartPC, method = "manb")
avNNet <- train(classe ~., data = trainPartPC, method = "avNNet")
mlp <- train(classe ~., data = trainPartPC, method = "mlp")
mlpWeightDecay <- train(classe ~., data = trainPartPC, method = "mlpWeightDecay")
mlpWeightDecayML <- train(classe ~., data = trainPartPC, method = "mlpWeightDecayML")
mlpML <- train(classe ~., data = trainPartPC, method = "mlpML")
mlpSGD <- train(classe ~., data = trainPartPC, method = "mlpSGD")
earth <- train(classe ~., data = trainPartPC, method = "earth")
gcvEarth <- train(classe ~., data = trainPartPC, method = "gcvEarth")
nb <- train(classe ~., data = trainPartPC, method = "nb")
nbDiscrete <- train(classe ~., data = trainPartPC, method = "nbDiscrete")
awnb <- train(classe ~., data = trainPartPC, method = "awnb")
pam <- train(classe ~., data = trainPartPC, method = "pam")
nnet <- train(classe ~., data = trainPartPC, method = "nnet")
pcaNNet <- train(classe ~., data = trainPartPC, method = "pcaNNet")
ORFlog <- train(classe ~., data = trainPartPC, method = "ORFlog")
ORFpls <- train(classe ~., data = trainPartPC, method = "ORFpls")
ORFridge <- train(classe ~., data = trainPartPC, method = "ORFridge")
ORFsvm <- train(classe ~., data = trainPartPC, method = "ORFsvm")
oblique.tree <- train(classe ~., data = trainPartPC, method = "oblique.tree")
ownn <- train(classe ~., data = trainPartPC, method = "ownn")
polr <- train(classe ~., data = trainPartPC, method = "polr")
parRF <- train(classe ~., data = trainPartPC, method = "parRF")
partDSA <- train(classe ~., data = trainPartPC, method = "partDSA")
kernelpls <- train(classe ~., data = trainPartPC, method = "kernelpls")
pls <- train(classe ~., data = trainPartPC, method = "pls")
simpls <- train(classe ~., data = trainPartPC, method = "simpls")
widekernelpls <- train(classe ~., data = trainPartPC, method = "widekernelpls")
plsRglm <- train(classe ~., data = trainPartPC, method = "plsRglm")
pda <- train(classe ~., data = trainPartPC, method = "pda")
pda2 <- train(classe ~., data = trainPartPC, method = "pda2")
PenalizedLDA <- train(classe ~., data = trainPartPC, method = "PenalizedLDA")
plr <- train(classe ~., data = trainPartPC, method = "plr")
multinom <- train(classe ~., data = trainPartPC, method = "multinom")
ordinalNet <- train(classe ~., data = trainPartPC, method = "ordinalNet")
qda <- train(classe ~., data = trainPartPC, method = "qda")
stepQDA <- train(classe ~., data = trainPartPC, method = "stepQDA")
rbf <- train(classe ~., data = trainPartPC, method = "rbf")
rbfDDA <- train(classe ~., data = trainPartPC, method = "rbfDDA")
rFerns <- train(classe ~., data = trainPartPC, method = "rFerns")
ranger <- train(classe ~., data = trainPartPC, method = "ranger")
Rborist <- train(classe ~., data = trainPartPC, method = "Rborist")
rf <- train(classe ~., data = trainPartPC, method = "rf")
extraTrees <- train(classe ~., data = trainPartPC, method = "extraTrees")
rfRules <- train(classe ~., data = trainPartPC, method = "rfRules")
Boruta <- train(classe ~., data = trainPartPC, method = "Boruta")
rda <- train(classe ~., data = trainPartPC, method = "rda")
rlda <- train(classe ~., data = trainPartPC, method = "rlda")
RRF <- train(classe ~., data = trainPartPC, method = "RRF")
RRFglobal <- train(classe ~., data = trainPartPC, method = "RRFglobal")
Linda <- train(classe ~., data = trainPartPC, method = "Linda")
rmda <- train(classe ~., data = trainPartPC, method = "rmda")
QdaCov <- train(classe ~., data = trainPartPC, method = "QdaCov")
rrlda <- train(classe ~., data = trainPartPC, method = "rrlda")
RSimca <- train(classe ~., data = trainPartPC, method = "RSimca")
rocc <- train(classe ~., data = trainPartPC, method = "rocc")
rotationForest <- train(classe ~., data = trainPartPC, method = "rotationForest")
rotationForestCp <- train(classe ~., data = trainPartPC, method = "rotationForestCp")
JRip <- train(classe ~., data = trainPartPC, method = "JRip")
PART <- train(classe ~., data = trainPartPC, method = "PART")
bdk <- train(classe ~., data = trainPartPC, method = "bdk")
xyf <- train(classe ~., data = trainPartPC, method = "xyf")
nbSearch <- train(classe ~., data = trainPartPC, method = "nbSearch")
sda <- train(classe ~., data = trainPartPC, method = "sda")
CSimca <- train(classe ~., data = trainPartPC, method = "CSimca")
C5.0Rules <- train(classe ~., data = trainPartPC, method = "C5.0Rules")
C5.0Rules
C5.0Tree <- train(classe ~., data = trainPartPC, method = "C5.0Tree")
OneR <- train(classe ~., data = trainPartPC, method = "OneR")
sdwd <- train(classe ~., data = trainPartPC, method = "sdwd")
sparseLDA <- train(classe ~., data = trainPartPC, method = "sparseLDA")
smda <- train(classe ~., data = trainPartPC, method = "smda")
spls <- train(classe ~., data = trainPartPC, method = "spls")
slda <- train(classe ~., data = trainPartPC, method = "slda")
snn <- train(classe ~., data = trainPartPC, method = "snn")
dnn <- train(classe ~., data = trainPartPC, method = "dnn")
sddaLDA <- train(classe ~., data = trainPartPC, method = "sddaLDA")
sddaQDA <- train(classe ~., data = trainPartPC, method = "sddaQDA")
gbm <- train(classe ~., data = trainPartPC, method = "gbm")
svmBoundrangeString <- train(classe ~., data = trainPartPC, method = "svmBoundrangeString")
svmRadialWeights <- train(classe ~., data = trainPartPC, method = "svmRadialWeights")
svmExpoString <- train(classe ~., data = trainPartPC, method = "svmExpoString")
svmLinear <- train(classe ~., data = trainPartPC, method = "svmLinear")
svmLinear2 <- train(classe ~., data = trainPartPC, method = "svmLinear2")
svmPoly <- train(classe ~., data = trainPartPC, method = "svmPoly")
svmRadial <- train(classe ~., data = trainPartPC, method = "svmRadial")
svmRadialCost <- train(classe ~., data = trainPartPC, method = "svmRadialCost")
svmRadialSigma <- train(classe ~., data = trainPartPC, method = "svmRadialSigma")
svmSpectrumString <- train(classe ~., data = trainPartPC, method = "svmSpectrumString")
tan <- train(classe ~., data = trainPartPC, method = "tan")
tanSearch <- train(classe ~., data = trainPartPC, method = "tanSearch")
awtan <- train(classe ~., data = trainPartPC, method = "awtan")
evtree <- train(classe ~., data = trainPartPC, method = "evtree")
nodeHarvest <- train(classe ~., data = trainPartPC, method = "nodeHarvest")
vbmpRadial <- train(classe ~., data = trainPartPC, method = "vbmpRadial")
wsrf <- train(classe ~., data = trainPartPC, method = "wsrf")



