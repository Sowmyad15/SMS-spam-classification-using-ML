
raw_sms <- read.csv("C:/Users/sowmy/OneDrive/Documents/Project/Proj/sms_spam.csv",stringsAsFactors = FALSE)
str(raw_sms)  

raw_sms$type <- factor(raw_sms$type)
str(raw_sms$type)
table(raw_sms$type)


# Cleaning the texts
library(NLP)
library(tm)
## Loading required package: NLP
library(SnowballC)

corpus = VCorpus(VectorSource(raw_sms$text))
print(corpus)
inspect(corpus[1:2])
as.character(corpus[[1]])

#DATA transformation
as.character(corpus[[1]])
corpus = tm_map(corpus, content_transformer(tolower))
as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers)
as.character(corpus[[1]])
corpus = tm_map(corpus, removePunctuation)
as.character(corpus[[1]])
corpus = tm_map(corpus, removeWords, stopwords("english"))
as.character(corpus[[1]])
corpus = tm_map(corpus, stemDocument)
as.character(corpus[[1]])
corpus = tm_map(corpus, stripWhitespace)
as.character(corpus[[1]])


sms_dtm <- DocumentTermMatrix(corpus)
sms_dtm

sms_dtm = removeSparseTerms(sms_dtm, 0.995)
dim(sms_dtm)

inspect(sms_dtm[40:50, 10:15])

dataset = as.data.frame(as.matrix(sms_dtm))
colnames(dataset) <- make.names(colnames(dataset))

dataset$type = raw_sms$type
str(dataset$type)
dataset$text


set.seed(222)
split = sample(2,nrow(dataset),prob = c(0.75,0.25),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

prop.table(table(train_set$type))

prop.table(table(test_set$type))

library(rpart)
library(rpart.plot)
dt_classifier <-rpart(type~., data = train_set, method = "class", minbucket = 35)

rpart.plot(dt_classifier, nn=TRUE)
dt_predict <- predict(dt_classifier, test_set, type = "class")



library(gmodels)
CrossTable(dt_predict, test_set$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
library(caret)
confusionMatrix(dt_predict,test_set$type)
