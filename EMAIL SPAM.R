
# Email Spam Detection Using Naive Bayes in R

# 1. Install and load required packages
if (!require(tm)) {
  install.packages("tm", dependencies = TRUE)
}
if (!require(e1071)) {
  install.packages("e1071", dependencies = TRUE)
}

library(tm)
library(e1071)

# 2. Create a sample email dataset
emails <- data.frame(
  text = c(
    "Win a free prize now",
    "Congratulations you won a lottery",
    "Claim your free gift today",
    "Limited offer win cash now",
    "Meeting scheduled for tomorrow",
    "Please submit your assignment",
    "Project discussion after lunch",
    "Your class timetable is updated",
    "Please attend the college meeting",
    "The assignment deadline is tomorrow",
    "Get a free mobile recharge",
    "Win exciting rewards today",
    "Team meeting at ten o clock",
    "Submit the project report today",
    "Congratulations claim your reward"
  ),
  category = factor(c(
    "spam", "spam", "spam", "spam",
    "not_spam", "not_spam", "not_spam",
    "not_spam", "not_spam", "not_spam",
    "spam", "spam", "not_spam",
    "not_spam", "spam"
  ))
)

# 3. Display the dataset
print("Email Dataset:")
print(emails)

# 4. Convert email text into a corpus
corpus <- VCorpus(VectorSource(emails$text))

# 5. Data preprocessing
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# 6. Create a Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)

# Convert Document-Term Matrix into a data frame
email_matrix <- as.data.frame(as.matrix(dtm))

# Add email categories
email_matrix$category <- emails$category

# 7. Split data into training and testing sets
set.seed(123)

train_index <- sample(
  seq_len(nrow(email_matrix)),
  size = floor(0.7 * nrow(email_matrix))
)

train_data <- email_matrix[train_index, ]
test_data <- email_matrix[-train_index, ]

# 8. Train the Naive Bayes classifier
model <- naiveBayes(
  category ~ .,
  data = train_data,
  laplace = 1
)

print("Naive Bayes Model:")
print(model)

# 9. Predict email categories
predictions <- predict(
  model,
  newdata = test_data,
  type = "class"
)

# 10. Display actual and predicted results
results <- data.frame(
  Actual = test_data$category,
  Predicted = predictions
)

print("Prediction Results:")
print(results)

# 11. Evaluate model performance
confusion_matrix <- table(
  Actual = test_data$category,
  Predicted = predictions
)

print("Confusion Matrix:")
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) /
  sum(confusion_matrix)

cat("Accuracy:", round(accuracy * 100, 2), "%\n")

# 12. Classify a new email
new_email <- "Congratulations win a free gift offer"

# Apply the same preprocessing steps
new_corpus <- VCorpus(VectorSource(new_email))
new_corpus <- tm_map(
  new_corpus, content_transformer(tolower)
)
new_corpus <- tm_map(new_corpus, removePunctuation)
new_corpus <- tm_map(new_corpus, removeNumbers)
new_corpus <- tm_map(
  new_corpus, removeWords, stopwords("en")
)
new_corpus <- tm_map(new_corpus, stripWhitespace)

# Create a Document-Term Matrix for the new email
new_dtm <- DocumentTermMatrix(
  new_corpus,
  control = list(dictionary = Terms(dtm))
)

new_data <- as.data.frame(as.matrix(new_dtm))

# Predict the category of the new email
new_prediction <- predict(
  model,
  newdata = new_data,
  type = "class"
)

cat("\nNew Email:", new_email, "\n")
cat("Predicted Category:", as.character(new_prediction), "\n")

# 13. Display final classification
if (new_prediction == "spam") {
  cat("Result: This email is SPAM.\n")
} else {
  cat("Result: This email is NOT SPAM.\n")
}

