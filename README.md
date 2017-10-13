# Movie-Review-Classification
Developed a  linear regression model with gradient descent along with tf-idf features for classification of movie reviews

Sentiment analysis is a common text categorization task. A popular method for learning
sentiment from a text document is through supervised machine learning techniques such as
linear regression. In this homework, you need to write a function with gradient descent
algorithm to learn the parameters of a linear regression model – this is also called training
phase of any supervised machine learning algorithm.

Data
The movie_reviews.csv contains sentences with their associated sentiment labels. The
sentiment labels are:
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
During the training phase, a learning algorithm takes documents as
inputs and does two things: 1) extracts features, and 2) learns weight/parameter of these
features. 
