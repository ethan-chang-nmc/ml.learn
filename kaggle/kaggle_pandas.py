import pandas as pd

'''
Creating, Reading, and Writing
'''
# Data frames
fruits = pd.DataFrame({"Apples": [30], "Bananas": [21]})
fruit_sales = pd.DataFrame({"Apples": [35, 41], "Bananas": [21,34]}, index = ["2017 Sales", "2018 Sales"])

# Series
ingredients = pd.Series({"Flour": "4 cups", "Milk": "1 cup", "Eggs": "2 large", "Spam": "1 can"}, name = "Dinner")

# Reading CSV -> DataFrame
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

# Reading DataFrame -> CSV
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
animals.to_csv("cows_and_goats.csv")


'''
Indexing, Selecting, and Assigning
'''
# Native accesors: let "reviews" be dataframe
reviews['country'][0]
reviews.country

# Indexing: loc and iloc (row-first, column second [opposite of native python])
reviews.iloc[:, 0] # all rows, column 0
reviews.iloc[1:3, 0]
reviews.iloc[[0, 1, 2], 0]
reviews.iloc[-5:] # last 5 elements
reviews.loc[0, 'country'] # loc is label based - iloc treats dataset like big matrix (list of lists) to index into postion while loc uses information in indices
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']] # [0:10] - iloc stdlib indexing (exclusive) but loc inclusive of the 10

# Manipulating Index: index is mutable
reviews.set_index("title")

# Conditional Selection
reviews.country == 'Italy' # returns 2 column index and boolean of is true or not (and Name: country, Length: 129971, dtype: bool)
reviews.loc[reviews.country == 'Italy'] # select relevant data
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)] # and
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)] # or
reviews.loc[reviews.country.isin(['Italy', 'France'])] # selects data with value in list of values
reviews.loc[reviews.price.notnull()] # aslo has isnull (empty, NaN)

# Assigning data
reviews['critic'] = 'everyone'
reviews['index_backwards'] = range(len(reviews), 0, -1)
