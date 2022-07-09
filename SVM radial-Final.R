
library(NLP)
library(tm)
library(SnowballC)
raw_sms <- read.csv("C:/Users/sowmy/OneDrive/Documents/Project/Proj/sms_spam.csv",stringsAsFactors = FALSE)
str(raw_sms)  

raw_sms$type <- factor(raw_sms$type)
str(raw_sms$type)
table(raw_sms$type)



# Cleaning the texts


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

dataset$type = raw_sms$type
str(dataset$type)
dataset$text


set.seed(222)
split = sample(2,nrow(dataset),prob = c(0.75,0.25),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

prop.table(table(train_set$type))

prop.table(table(test_set$type))

library(e1071)
svm_classifier <- svm(type~., train_set,kernel="radial")
sms_test_pred <- predict(svm_classifier, test_set)
head(sms_test_pred)

library(gmodels)
CrossTable(sms_test_pred, test_set$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
library(caret)
confusionMatrix(sms_test_pred,test_set$type)
