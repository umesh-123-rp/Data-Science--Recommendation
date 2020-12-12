#title: "Book Recommender"
#subtitle: "Exploratory Analysis & Collaborative Filtering

#This external dataset allows us to take a deeper look at data-driven book recommendations.
# The analysis can be divided into three parts:
  
# Part I:  Explores the dataset to find some interesting insights.  
# Part II: Introduces and demonstrates collaborative filtering .
# Part III:Creates a functioning book recommender(to recommend some books)

## Part I: Exploratory Analysis 

# We start by loading some libraries and reading the data file:
library(recommenderlab)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(DT)
library(knitr)
library(grid)
library(gridExtra)
library(corrplot)
library(qgraph)
library(methods)
library(Matrix)

### Read the file
ratings <- fread("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Recommendation System-R\\bookre.csv")

# First, let's have a look at the data set ratings.csv

View(ratings)

# The file `ratings.csv` contains all user's ratings of the books(a total of 10,000 books) 

### Clean the dataset
# As with nearly any real-life dataset, we need to do some cleaning first. 
# While exploring the data,We have noticed that for some combinations of user and book 
# there are multiple ratings, while in theory there should only be one unless it is allowed.
# So let's first remove the duplicate ratings.
ratings[, N := .N, .(User.ID, Book.Title)]
cat('Number of duplicate ratings: ', nrow(ratings[N > 1]))
ratings <- ratings[N == 1]
# There are 14 numbers of Duplicate ratings which need to be removed.
#Furthermore,it is better to have more ratings per user for collaborative filtering.
#So I decided to remove users who have rated fewer than 3 books. 
#Let's remove users who rated fewer than 3 books. 
ratings[, N := .N, .(User.ID)]
cat('Number of users who rated fewer than 3 books: ', uniqueN(ratings[N <= 2, User.ID]))
ratings <- ratings[N > 2]
# We have found that there are 1571 users who have not rated more than 3 books
# We will not consider them for our analysis

### Let's start the Exploration
#What is the distribution of ratings?
#We see that people tend to give quite positive ratings to books. 
# Most of the ratings are in the 5-10 range, while very few ratings are in the 1-4 range. 
ratings %>% 
  ggplot(aes(x =Book.Rating, fill = factor(Book.Rating))) +
  geom_bar(color = "grey20") + scale_fill_brewer(palette = "YlGnBu") + guides(fill = FALSE)

### Number of ratings per user
# As we filtered our ratings all users have at least 3 ratings.
# However, we can also see that are some users with many ratings. 
# This is interesting, because we can later examine whether frequent raters rate books differently from less frequent raters. We will come back to this later. 
ratings %>% 
  group_by(User.ID) %>% 
  summarize(number_of_ratings_per_user = n()) %>% 
  ggplot(aes(number_of_ratings_per_user)) + 
  geom_bar(fill = "cadetblue3", color = "grey20") + coord_cartesian(c(3, 50))
# There are many users who have rated less no. of books and few users who have rated more than 50 books
### Distribution of mean user ratings
ratings %>% 
  group_by(User.ID) %>% 
  summarize(mean_user_rating = mean(Book.Rating)) %>% 
  ggplot(aes(mean_user_rating)) +
  geom_histogram(fill = "cadetblue3", color = "grey20")
# The mean rating f users are close to 7 and 8

### Number of ratings per book
#We can see that in the subsetted dataset most books have around 1-2 ratings. 
ratings %>% 
  group_by(Book.Title) %>% 
  summarize(number_of_ratings_per_book = n()) %>% 
  ggplot(aes(number_of_ratings_per_book)) + 
  geom_bar(fill = "orange", color = "grey20", width = 1) + coord_cartesian(c(0,4))

#### Distribution of mean book ratings
# Mean book ratings don't reveal any peculiarities. 
ratings %>% 
  group_by(Book.Title) %>% 
  summarize(mean_book_rating = mean(Book.Rating)) %>% 
  ggplot(aes(mean_book_rating)) + geom_histogram(fill = "orange", color = "grey20") + coord_cartesian(c(1,10))

### Do frequent raters rate differently?

# It is possible, that users of frequent raters rate differently from less frequent raters.
# Explores this possibility. It seems like frequent raters tend to give lower ratings to books,
# maybe they are more critical the more they read and rate. That's interesting. 

get_cor <- function(df){
  m <- cor(df$x,df$y, use="pairwise.complete.obs");
  eq <- substitute(italic(r) == cor, list(cor = format(m, digits = 2)))
  as.character(as.expression(eq));                 
}

temp <-ratings %>% 
  group_by(User.ID) %>% 
  summarize(mean_rating = mean(Book.Rating), number_of_rated_books = n())

temp %>% filter(number_of_rated_books <= 100) %>% 
  ggplot(aes(number_of_rated_books, mean_rating)) + stat_bin_hex(bins = 50) + scale_fill_distiller(palette = "Spectral") + stat_smooth(method = "lm", color = "orchid", size = 2, se = FALSE) +
  annotate("text", x = 80, y = 1.9, label = get_cor(data.frame(x = temp$number_of_rated_books, y = temp$mean_rating)), color = "orchid", size = 7, parse = TRUE)

### Summary - Part I

# We identified some interesting aspects of this book datasets. 
# In summary, observed effects on book rating are rather small, 
# suggesting that book rating is mainly driven by other aspects, 
# hopefully including the quality of the book itself. 
# In part II we are going to look at collaborative filtering and 
# eventually build a recommender system

### Part II: Collaborative Filtering

# Collaborative filtering is a standard method for product recommendations.
# user-based collaborative filtering. It works as follows:  
  
# 1.First identify other users similar to the current user in terms of 
#   their ratings on the same set of books.  

# 2.Take Average rating of similar users of books the current user has not yet read ...  
 
# 3.Recommend those books with the highest average rating to the current user.  

# These three steps can easily be translated into an alogrithm. 

#However, before we can do that we have to restructure our data.
#For collaborative filtering data are usually structured that each row 
#corresponds to a user and each column corresponds to a book.
#To restructure our rating data from this dataset in the same way,
#we can do the following:

dimension_names <- list(User.ID= sort(unique(ratings$User.ID)), Book.Title = sort(unique(ratings$Book.Title)))
ratingmat <- spread(select(ratings, Book.Title, User.ID, Book.Rating), Book.Title, Book.Rating) %>% select(-User.ID)

ratingmat <- as.matrix(ratingmat)
dimnames(ratingmat) <- dimension_names
ratingmat[1:5, 1:5]
dim(ratingmat)

#We see that our rating matrix has 611 rows x 7933 columns.
# It's time to go through our 3 steps:

### Step 1: Find similar users

# For this step we select users that have in common that they rated the same books. 
# To make it easier let's select one example user.id:507. 
# First we select users that rated at least one book that the user 507 has also rated. 

current_user <- "507"
rated_items <- which(!is.na((as.data.frame(ratingmat[current_user, ]))))
selected_users1 <- names(which(apply(!is.na(ratingmat[ ,rated_items]), 1, sum) >= 1))
head(selected_users1, 4)
user1 <- data.frame(item=colnames(ratingmat),rating=ratingmat["507",]) %>% filter(!is.na(rating))

#For these users, we can calculate the similarity of their ratings with a current user rating
#There is a number of options to calculate similarity. 
#Typically cosine similarity or pearson's correlation coefficient are used. 
#We would now go through all the selected users and calculate the similarity between their and the current user ratings. Below
#I do this for 2 users (User.ID: 507 and 2439) for illustration. 
#We can see that similarity is higher for user 507 than user 2439 

user1 <- data.frame(item=colnames(ratingmat),rating=ratingmat[current_user,]) %>% filter(!is.na(rating))
user2 <- data.frame(item = colnames(ratingmat),rating=ratingmat["2439",]) %>% filter(!is.na(rating))

tmp<-merge(user1, user2, by="item")
tmp
cor(tmp$rating.x, tmp$rating.y, use="pairwise.complete.obs")

#To reduce the influence of inter individual differences in mean ratings, We can normalize. 
rmat <- ratingmat[selected_users1, ]
user_mean_ratings <- rowMeans(rmat,na.rm=T)
rmat <- rmat - user_mean_ratings

#We can calculate the similarity of all others users with the current user
#and sort them according to the highest similarity.

similarities <- cor(t(rmat[rownames(rmat)!=current_user, ]), rmat[current_user, ], use = 'pairwise.complete.obs')
sim <- as.vector(similarities)
names(sim) <- rownames(similarities)
res <- sort(sim, decreasing = TRUE)
head(res,4)


# We can now select the 3 most similar users: 625, 2439 and 3728

#### Visualizing similarities between users
# Similarities between users can be visualized using the qgraph package. 
# The width of the graph's edges correspond to similarity 
# (blue for positive correlations, red for negative correlations).

sim_mat <- cor(t(rmat), use = 'pairwise.complete.obs')
random_users <- selected_users1[1:3]
qgraph(sim_mat[c(current_user, random_users), c(current_user, random_users)],
layout = "spring", vsize = 5, theme = "TeamFortress", labels = c(current_user, 
                                                                                                   random_users))

### Using recommenderlab -PART III - Last part to recommend the right book/ books

#Recommenderlab is a R-package that provides the infrastructure to evaluate and
#compare several collaborative-filtering algortihms. 
# We can represent of our rating matrix in the form of SparseMatrix
ratingmat0 <- ratingmat
ratingmat0[is.na(ratingmat0)] <- 0
sparse_ratings <- as(ratingmat0, "sparseMatrix")
rm(ratingmat0)
gc()

#Recommenderlab uses as special variant of a sparse matrices, so we convert to this class first.

real_ratings <- new("realRatingMatrix", data = sparse_ratings)
real_ratings


#Running an algorithm in Recommenderlab with method "UBCF"- User Based Collaborative Filter 

model <- Recommender(real_ratings, method = "UBCF", param = list(method = "pearson", nn = 4))


#Creating predictions ratings 
## create n recommendations for a user based on his ratings

uid = "507"
books <- subset(ratings, ratings$User.ID==uid)
print("You have rated:")
books

#Prediction For User
print("Recommendations for you:")
prediction <-predict(model, real_ratings[uid], n = 5) 
as(prediction, "list")
# UBCF is not working in this case as the data size is very large.

### Let's apply Different techniques
#Popularity based 

model1 <- Recommender(real_ratings, method="POPULAR")

#Predictions for user
recommended_items1 <- predict(model1, real_ratings[uid], n=5)
as(recommended_items1, "list")

#Singular value decomposition (SVD) for dimensionality reduction and 
#construct the matrices with row of users and columns of items

model2 <- Recommender(real_ratings, method="SVD")

#Predictions for user
recommended_items2 <- predict(model2, real_ratings[uid], n=5)
as(recommended_items2, "list")


#We see that recommendations are nearly the same compared to our basic 
#algorithm described above. The top 4 recommended books are exactly the same ones. 


#### Evaluating the predictions

# Let's create an evaluation scheme with 10-fold cross validation. 
#In this case -1 means that the predictions are calculated from all but 
# 1 ratings, and performance is evaluated for 1  for each user. 

scheme <- evaluationScheme(real_ratings[1:500,], method = "cross-validation", k = 10, given = -1, goodRating = 5)

#Let's compare different techniques with different value w.r.t their RMSE values
#Furthermore, as a baseline one can also add an algorithm ("RANDOM") 

algorithms <- list("random" = list(name = "RANDOM", param = NULL),
                   "UBCF_05" = list(name = "UBCF", param = list(nn = 5)),
                   "UBCF_10" = list(name = "UBCF", param = list(nn = 10)),
                   "UBCF_30" = list(name = "UBCF", param = list(nn = 30)),                   
                   "UBCF_50" = list(name = "UBCF", param = list(nn = 50))
                   )
# evaluate the alogrithms with the given scheme            
results <- evaluate(scheme, algorithms, type = "ratings")
# It is observed that the UBCF is failing to work in this case due to large data size
# Therefore the following methods were tried out

recommenderRegistry$get_entry_names()

#You can get more information about these algorithms:

recommenderRegistry$get_entries(dataType = "realRatingMatrix")

# Let's use other two additional algorithms, "popular" and "SVD" to predicts ratings 

scheme <- evaluationScheme(real_ratings[1:500,], method = "cross-validation", k = 10, given = -1, goodRating = 5)

algorithms <- list("random" = list(name = "RANDOM", param = NULL),
                   "popular" = list(name = "POPULAR"),
                   "SVD" = list(name = "SVD")
                   )
                   
results <- evaluate(scheme, algorithms, type = "ratings", progress = FALSE)

# restructure results output
tmp <- lapply(results, function(x) slot(x, "results"))
res <- tmp %>% 
  lapply(function(x) unlist(lapply(x, function(x) unlist(x@cm[ ,"RMSE"])))) %>% 
  as.data.frame() %>% 
  gather(key = "Algorithm", value = "RMSE")

res %>% 
  mutate(Algorithm=factor(Algorithm, levels = c("random", "popular", "SVD"))) %>%
  ggplot(aes(Algorithm, RMSE, fill = Algorithm)) + geom_bar(stat = "summary") + 
  geom_errorbar(stat = "summary", width = 0.3, size = 0.8) + coord_cartesian(ylim = c(0.6, 1.3)) + 
  guides(fill = FALSE)

### CONCLUSION
# UBCF is not working well with the large dataset
# RANDOM, POPULAR AND SVD algorithms are working with same value of RMSE.
# Therefore, we can adopt any one of the method for prediction 
