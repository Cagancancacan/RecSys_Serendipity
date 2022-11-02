# RecSys_Serendipity
Group Serendipity for Recommender Systems

## Group Members
Can Idil (6194134), \
Roman Ilic (6253448), \
Jack Waterman (6262576) 

## Individual Recommender System

The dataset used for this recommender system can be found at: https://files.grouplens.org/datasets/movielens/ml-20m.zip

### How to Run:
Inside the individual_recommender folder 
1. Create a ml-20m folder.
2. Extract the ml-20m into this folder.
3. Create a dataset folder and create the following .csv files 
   1. average_ratings.csv
   2. data.csv 
   3. expert_users.csv 
   4. genre.csv 
   5. movies.csv 
   6. ratings.csv 
   7. ratings_genres.csv 
   8. tags.csv
4. Run the Content_Preprocessing.ipynb notebook.
5. Then either the Content_Notebook.ipynb or Content_Rec_Sys.py to generate movie recommendations for a random expert user.


## Group Recommender System

### How to Run:
1. Reuse the previously used data.csv file (average ratings per user for each genre)
2. Specify the number of clusters (groups) to generate (adapt code if necessary)
3. Run the rest of the Group_Recommendation.ipynb notebook

*note:
- in our case, group recommendation have been done for one group only (group 1)
- prediction for a user and a specific movie have been set to the user's average rating to the genre of that movie (average if several genres)
- once aggregation matrix complete, fairness have been applied 

