import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

''' Created a filtered data set
data = pd.read_csv("just tacos and burritos.csv")
filtered_data = data.dropna(axis='columns', how='all')
filtered_data.to_csv('filteredTBdata.csv')
'''
data = pd.read_csv("filteredTBdata.csv", index_col=0, header=0)
numCategories = 4
numCategories = numCategories + 1

''' Checking for and dropping duplicates
temp_df = data.append(data)
temp_df = temp_df.drop_duplicates()
print(data.shape)
print(temp_df.shape)
'''

#Data cleanup: drop extraneous columns and filter out meat options
data.drop(['id', 'keys', 'menus.dateSeen', 'priceRangeCurrency', 'priceRangeMin', 'menuPageURL',
 'menus.amountMin', 'menus.category', 'dateAdded', 'dateUpdated', 'websites',
 'address', 'country', 'menus.currency', 'postalCode'], axis=1, inplace=True)
data = data[pd.notnull(data['latitude'])]

data = data[~data['menus.name'].str.contains('Shrimp') & ~data['menus.name'].str.contains('Beef')
 & ~data['menus.name'].str.contains('Breakfast') & ~data['menus.name'].str.contains('Chicken')
 & ~data['menus.name'].str.contains('Sausage') & ~data['menus.name'].str.contains('Pork')
 & ~data['menus.name'].str.contains('Fish') & ~data['menus.name'].str.contains('Steak')
 & ~data['menus.name'].str.contains('Calamari') & ~data['menus.name'].str.contains('Egg')
 & ~data['menus.name'].str.contains('Pollo') & ~data['menus.name'].str.contains('Carne')
 & ~data['menus.name'].str.contains('Tilapia') & ~data['menus.name'].str.contains('Octopus')
 & ~data['menus.name'].str.contains('Bacon') & ~data['menus.name'].str.contains('Lobster')]

#print(data['longitude'].head())

texas_cities = data[(data['city'] == 'Dallas') | (data['city'] == 'Austin') | (data['city'] == 'Houston')]

#nullPositions = data.isnull()
#print("Null values: ")
#print(data.isnull().sum())

#Get mealCounts, which has the number of vegetarian options by city
mealCounts = data.groupby(['city', 'longitude', 'latitude']).count()
mealCounts.reset_index(level=[0,1,2], inplace=True)
mealCounts = mealCounts[['city', 'menus.name', 'latitude', 'longitude']]
aggregation_functions = {'longitude' : 'first', 'latitude' : 'first', 'menus.name' : 'sum'}
mealCounts = mealCounts.groupby('city', as_index = False).aggregate(aggregation_functions)
mealCounts.sort_values(by=['menus.name'], ascending = False, inplace=True)
mealCounts = mealCounts[mealCounts['menus.name'] > 5]
print(mealCounts.head())


#Run K means clustering to get good category values for the bubble map
input = mealCounts['menus.name'].to_numpy().reshape(-1, 1)
kmeans = KMeans(n_clusters = numCategories, random_state=0).fit(input)

categories = pd.Series(kmeans.predict(mealCounts['menus.name'].to_numpy().reshape(-1, 1)))
mealCounts.insert(4, "category", categories.tolist(), True)


#Configure and show the map
df = mealCounts

df['text'] = df['city'] + '<br>Vegetarian Tacos and Burritos: ' + (df['menus.name']).astype(str)
#limits = [(0,2), (3,10), (11,20),(21,50),(50,3000)]
ranges = [(21, 50), (16, 20), (11, 15), (5, 10)]
categories = range(1,numCategories)
colors = ['royalblue', 'crimson', 'lightseagreen', 'orange', 'lightgrey']
cities = []
scale = 1

fig = go.Figure()

for i in range(len(categories)):
    category = categories[i]
    df_sub = df[df['category'] == category]

    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = df_sub['text'],
        marker = dict(
            size= df_sub['menus.name'] * 1.5 ,
            color=colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode='area'
        ),
        name = '{0}'.format(kmeans.cluster_centers_[i + 1])
    ))

    fig.update_layout(
        title_text = 'Vegetarian Tacos and Burritos in the US',
        showlegend = True,
        geo = dict(
            scope='usa',
            landcolor='lightgray',
            #lonaxis_range=[-100.0, -90.0]
        )
    )

fig.show()

'''
Ideas for improvements
    -Put the data in a tabular format (maybe top 10 cities)
    -Get the meat data equivalent to find discrepancies
    -Create a static website to display the data
    -Use the description entries to get rid of more meat options


'''