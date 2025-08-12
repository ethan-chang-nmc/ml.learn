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


'''
Summary Functions and Maps
'''
# Summary Functions (not offical name): restructures data in a useful way
reviews.points.describe() # type-aware, numerics(count, mean, max, quartiles) string(count, uniqe, top, freq)
reviews.points.mean()
reviews.taster_name.unique()
reviews.taster_name.value_counts()

# Maps: returns new transformed Series/DataFrames, original data not modified
review_points_mean = reviews.points.mean() # remean to 0
reviews.points.map(lambda p: p - review_points_mean) # remean to 0, map() expects single value from series and returns transformed
def remean_points(row):
    row.points = row.points - review_points_mean
    return row
reviews.apply(remean_points, axis='columns') # apply function to each row (axis tells it to grab a series of all columns -> creates row)
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean # faster way to remean due to speed ups built into pandas
reviews.country + " - " + reviews.region_1 # concatenate Series of equal length, standard operators work in this manner but less flexible than map() or apply()

# Problems and solutions
# What is the median of the points column in the reviews DataFrame?
median_points = reviews.points.median()

# What countries are represented in the dataset? (Your answer should not include any duplicates.)
countries = reviews.country.unique()

# How often does each country appear in the dataset? Create a Series reviews_per_country mapping countries to the count of reviews of wines from that country.
reviews_per_country = reviews.country.value_counts()

# Create variable centered_price containing a version of the price column with the mean price subtracted.
# (Note: this 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.)
centered_price = reviews.price - reviews.price.mean()

# I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable bargain_wine with the title of the wine with the highest points-to-price ratio in the dataset.
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

# There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series descriptor_counts counting how many times each of these two words appears in the description column in the dataset. (For simplicity, let's ignore the capitalized versions of these words.)
trop_count = reviews.description.map(lambda desc: "tropical" in desc).sum()
fruity_count = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([trop_count, fruity_count], index=['tropical', 'fruity'])

# We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.
# Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.
# Create a series star_ratings with the number of stars corresponding to each review in the dataset.
def star(row):
    if (row.country == "Canada") or (row.points >= 95):
        return 3
    elif (row.points >= 85) and (row.points < 95):
        return 2
    else:
        return 1
star_ratings = reviews.apply(star, axis="columns")


'''
Grouping and Sorting
'''
# Groupwise Analysis
reviews.groupby('points').points.count() # replicates what value_counts() does by creating group of reviews w/ same point and then counting how many appearances
reviews.groupby('points').price.min()
reviews.groupby('winery').apply(lambda df: df.title.iloc[0]) # groups are thought of as a slice of DF containing values that match, use apply() to manipulate data (ex retribes first wine from each winery)
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]) # finer control
reviews.groupby(['country']).price.agg([len, min, max]) # runs multiple functions on DataFrame simultaniously

# Multi-indexes
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])