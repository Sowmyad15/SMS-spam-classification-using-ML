#Using NB for SMS spam classification
#Explore the data set
#import the data set

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

#word frequency
freq<- sort(colSums(as.matrix(sms_dtm)), decreasing=TRUE)
findFreqTerms(sms_dtm, lowfreq=60)
head(freq)
library(ggplot2)
wf<- data.frame(word=names(freq), freq=freq)
head(wf)
pp <- ggplot(subset(wf, freq>100), aes(x=reorder(word, -freq), y =freq)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x=element_text(angle=45, hjust=1))
pp
library("wordcloud")
library("RColorBrewer")
set.seed(1234)
wordcloud(words = wf$word, freq = wf$freq, min.freq=1, max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))


convert_counts <- function(x) {
  x <- ifelse(x>0, "Yes", "No")
}

dtm_final <- apply(sms_dtm, 2, convert_counts)
dataset = as.data.frame(as.matrix(dtm_final))

dataset$type = raw_sms$type






set.seed(222)
split = sample(2,nrow(dataset),prob = c(0.75,0.25),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

prop.table(table(train_set$type))

prop.table(table(test_set$type))

library(e1071)
sms_classifier <- naiveBayes(train_set, train_set$type)
head(sms_classifier)
sms_test_pred <- predict(sms_classifier, test_set)
head(sms_test_pred)

library(gmodels)
CrossTable(sms_test_pred, test_set$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
library(caret)
confusionMatrix(sms_test_pred,test_set$type)


