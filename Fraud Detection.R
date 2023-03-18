install.packages("naivebayes")
install.packages('caTools')
install.packages("tidymodels")
install.packages("parsnip")
install.packages("brotools")
install.packages("mlbench")

library(tidyverse)
library(caret)
library(tm)
library(ggplot2)
library(e1071)
library(tidymodels)
library(parsnip)
library(brotools)
library(mlbench)
library(caTools)
library(naivebayes)
library(dplyr)

# Importing  and explore data
df <- read.csv("2tp_dataset_edited.csv")
str(df)
summary(df)
dim(df)
head(df)
tail(df)
names(df)
class(df)
View(df) 

#df[which(is.na(as.numeric(df[]))), 0]

#Remove unneccessary attributes
df$isFlaggedFraud <- NULL
df$oldbalanceOrg <- NULL
df$newbalanceOrig <- NULL
df$oldbalanceDest <- NULL
df$newbalanceDest <- NULL

summary(df)
str(df)

#Convert factors
df$type <- as.factor((df$type))
df$nameDest <- as.factor((df$nameDest))
df %>% str()

df$type %>% str()
df$type %>% summary()

df$nameDest %>% str()
df$nameDest %>% summary()

#Convert as numeric
df$type <- as.numeric((df$type))
df$nameDest <- as.numeric((df$nameDest))
df$isFraud <- as.numeric((df$isFraud))
#df$isFlaggedFraud <- as.numeric((df$isFlaggedFraud))
df %>% str()

#df$type %>% table()
#df$isFraud %>% table()
#df$isFlaggedFraud %>% table()
#df$nameDest %>% table()

# Encoding the target feature as factor
df$isFraud = factor(df$isFraud, levels = c(0, 1))
summary(df)
df %>% str()

typeof(df$isFraud)

df$isFlaggedFraud = factor(df$isFlaggedFraud, levels = c(0, 1))
summary(df)
df %>% str()

## set the seed to make your partition reproducible
set.seed(123) # This ensure the random result are reproduceable
trainIndex <- createDataPartition(df$isFraud, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Fitting Naive Bayes to the Training set
NBclassifier = naiveBayes(x = train[-4],
                        y = train$isFraud)

#naiveBayes.formula(formula = isFraud ~ ., data = df)

# Predicting the Test set results
y_pred = predict(NBclassifier, newdata = test[-4])

#DO NOT INCLUDE
# Making the Confusion Matrix
cm = table(test[, 4], y_pred)
head(cm)
#________________________________

# Compare result with confusion matrix
confusionMatrix(test$isFraud, y_pred, positive="1")

table(test[, 4], y_pred)

posPredValue(test$isFraud, y_pred, positive="1")

df$isFraud <- as.numeric((df$isFraud))
sensitivity(test$isFraud, y_pred, positive="1")

# Visualising the Training set results
install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
#split = sort(sample(nrow(df), nrow(df)*0.8))
split <- sample.split(df$isFraud, SplitRatio = 0.75)
train_test_split(df, test_proportion = 4/5)

training_set <-df[split,]
test_set <-df[-split,]

# Feature Scaling
training_set = scale(training_set[-8])
test_set[-8] = scale(test_set[-8])


#Plots and Correlations
pairs(df[0:4]) # error
cor(df[0:4]) # error



min(df$amount)
median(df$amount)
max(df$amount)



cor(df[complete_cases,2:5])

Sex_table <- table(titanic$Sex)
Sex_table

Sex_pct <-prop.table(Sex_table) * 100
Sex_pct
round(Sex_pct, digits = 1)

max(titanic$Fare)

data <- titanic
data %>% str()

complete_cases <- complete.cases(data)
complete_cases %>%str()

cor(data[complete_cases,2:5])

barplot(table(titanic$Survived, titanic[,3]))
legend("topleft", legend = c("Died", "Survived"), fill=c("black","grey"))

m = round(mean(titanic$Age, na.rm = TRUE),0) #mean calculation
titanic$Age <- replace(titanic$Age, is.na(titanic$Age), m)

dim(titanic)

train = titanic[1: 712,]
test = titanic[713:891,]

NBclassifier = naiveBayes(Survived ~ ., data = train)
NBclassifier

predictions <- predict(NBclassifier, test)
predictions

library(caret)

confusionMatrix(test$Survived, predictions)


# "%>%": this is called a "pipe", it puts the value on its left 
# to the first parameter of the functions on its right

df %>% str() #getting summary info of the df
df %>% head(5) # view the first 5 rows of the df

#Drop unnecessary column using the select function
#df<- df %>% select(-X)

#Convert label to factor data type using mutate
df<- df %>% mutate(label = as.factor(label))

df %>% str() #getthing summary info of the df again

# First, we will turn the SMS Column into a corpus
# This corpus allow us to work with every SMS of the columns at once

sms_corpus <- df$SMS %>% VectorSource() %>% VCorpus
sms_corpus #Lets you see the corpus

# You can use the as.character() function to read an SMS from the corpus
sms_corpus[[1]] %>% as.character()  # remember to use double bracket to indexing

# Creating the bag of word with DocumentTermMatrix() function
sms_dtm <- sms_corpus %>% DocumentTermMatrix(control =list(tolower = TRUE,
                                            removeNumbers = TRUE,
                                            stopwords = TRUE,
                                            removePunctuation = TRUE,
                                            stemming = TRUE))
sms_dtm

# Train test split
set.seed(520) # This ensure the random result are reproduceable
# createDataPartition generate random index values for a subset of a given column
train <- createDataPartition(y = df$label, p = 0.7, list = FALSE)

sms_dtm_train <- sms_dtm[train,]
sms_dtm_test <- sms_dtm[-train,]

df_train <- df[train,]
df_test <- df[-train,]

nrow(sms_dtm_train)
nrow(sms_dtm_test)

#findFreqTerms returns a set of most frequest word in the matrix
# We can set the lowest frequency for the function
sms_freq_words <- sms_dtm_train %>% findFreqTerms(lowfreq = 5)
head(sms_freq_words, 5)

# Select the columns with only frequent words by indexing 
sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test<- sms_dtm_test[ , sms_freq_words]

# Turning matrix into a Binary matrix
# Create function then apply it to all value of the matrix
feature_binarizer <- function(x){x <- ifelse (x > 0, 1 , 0)}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, feature_binarizer)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, feature_binarizer)

#Shows the matrix dimensions
dim(sms_train)
dim(sms_test)

# Modeling
set.seed(520)

# Evaluating model
# Run the prediction
predicted_label <-predict(nb_model, newdata = sms_test)

# Compare result with confusion matrix
confusionMatrix(predicted_label, df_test$label)


