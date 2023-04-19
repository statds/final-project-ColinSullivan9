import pandas as pd
from geopy.geocoders import Nominatim
from uszipcode import SearchEngine
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import stats
import numpy as np
print('\n\n\n\n\n\n\n\n\n\n\n\n\n')

df = pd.read_csv('Affordable_Housing_by_Town_2011-2022.csv')
df.columns = df.columns.str.replace(' ', '_')
# print(df.head())


## Find columns with the most missing data
missing_counts = df.isnull().sum()
# print(missing_counts)

## Data is pretty clean! Only one row has a missing value. Let's find that row and inspect it

mask = df.isnull().any(axis=1)

# select the rows with missing values
rows_with_missing_values = df[mask]

# print(rows_with_missing_values)

## Hmm. We may be able to fix this. Let's see if deed_restricted and government_restriced sum up to
## total_restricted

for index, row in df.iterrows():
    if (row["Deed_Restricted_Units"] + row["Government_Assisted"] + row["_Single_Family_CHFA/_USDA_Mortgages"] + row["Tenant_Rental_Assistance"]) != row["Total_Assisted_Units"]:
        print('Unsuccessful row:' + str(index))
        

## Success! We can now fill in the missing value appropriately

df.loc[530, "Government_Assisted"] = df.loc[530, "Total_Assisted_Units"] - df.loc[530, "Deed_Restricted_Units"] - df.loc[530, "_Single_Family_CHFA/_USDA_Mortgages"] - df.loc[530, "Tenant_Rental_Assistance"]
# print(df.loc[530])

for index, row in df.iterrows():
    if (row["Deed_Restricted_Units"] + row["Government_Assisted"] + row["_Single_Family_CHFA/_USDA_Mortgages"] + row["Tenant_Rental_Assistance"]) != row["Total_Assisted_Units"]:
        print('Unsuccessful row:' + str(index))
        
        
missing_counts = df.isnull().sum()
# print(missing_counts)



# 
# import os
# 
# # Define the path to the shapefile
# path_to_shapefile = "Desktop/DS_Final/ct_towns.shp"
# 
# # Check if the shapefile exists
# if os.path.exists(path_to_shapefile):
#     print("The file exists at the specified path.")
# else:
#     print("The file does not exist at the specified path.")



















## Success.
df_2017 = df[df.Year == 2017]
df_2017 = df_2017.rename(columns={'Town': 'town'})

df_2011 = df[df.Year == 2011]
df_2011 = df_2011.rename(columns={'Town': 'town'})

df_2017['2011_2017_change'] = ''



for index, row in df_2017.iterrows():
    mask = (df['Town'] == row['town']) & (df['Year'] == 2011)
    mask_row = df.loc[mask, 'Percent_Affordable'].values[0]
    df_2017.loc[index, '2011_2017_change'] = float(row['Percent_Affordable'] - mask_row)



# print('max:', df_2017['2011_2017_change'].max())
# 
# print('min:', df_2017['2011_2017_change'].min())

ct_boundary = gpd.read_file('geo_export_a270973b-90da-44a4-84c5-25a89abe61fb.shp')

# print(gdf_towns.columns)


# join the affordable housing data with the town shapefile on the Town column
ct_towns17 = ct_boundary.merge(df_2017, on='town', how='left')
ct_towns11 = ct_boundary.merge(df_2011, on='town', how='left')
ct_towns17['2011_2017_change'] = ct_towns17['2011_2017_change'].astype(float)
# 
# gdf_centroids = gdf_towns.copy()
# gdf_centroids['geometry'] = gdf_centroids['geometry'].centroid

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
# ct_towns.plot(column='Percent_Affordable', cmap='YlOrRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax)
# ax.set_aspect('equal')
ct_towns11.plot(column='Percent_Affordable', cmap='PuRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax1)
ax1.axis('off')
ax1.set_title('Percent Affordable Housing by\nTown in Connecticut, 2011')


# plot the centroids of each town on the map
# ct_boundary.plot(color='lightgray', edgecolor='white', linewidth=0.5)
ct_towns17.plot(column='Percent_Affordable', cmap='PuRd', linewidth=0.2, edgecolor='white', legend=True, ax=ax2)
ax2.axis('off')
ax2.set_title('Percent Affordable Housing by\nTown in Connecticut, 2017')

# ct_towns.plot(column='Percent_Affordable', cmap='YlOrRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax)
# ax.set_aspect('equal')

plt.tight_layout()
plt.show()

# df['2011_2017_change'] = df[]

fig, ax = plt.subplots(figsize=(10, 5))
ct_towns17.plot(column='2011_2017_change', cmap='PuOr', linewidth=0.2, edgecolor='white', legend=True, ax=ax)
ax.axis('off')
ax.set_title('Change in Percent Affordable Housing\nby Town in Connecticut, 2011-2017')
plt.show()





df_2011 = pd.read_csv('2011_crime.csv')
df_2012 = pd.read_csv('2012_crime.csv')
df_2013 = pd.read_csv('2013_crime.csv')
df_2014 = pd.read_csv('2014_crime.csv')
df_2015 = pd.read_csv('2015_crime.csv')
df_2016 = pd.read_csv('2016_crime.csv')
df_2017 = pd.read_csv('2017_crime.csv')

df_2011['Year'] = 2011
df_2012['Year'] = 2012
df_2013['Year'] = 2013
df_2014['Year'] = 2014
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017







ct_df_2017 = ct_towns17.merge(df_2017, on='town', how='left')
# print(ct_df_2017.columns)


# print(df_2011.iloc[2-174]['violent_crime'])
lst = [i for i in range(2, 174)]
test_mask = df_2017.iloc[lst, :]

# print(test_mask.violent_crime)
fig, (ax3, ax4) = plt.subplots(ncols=2, figsize=(10, 5))
ax3.set_aspect('equal')
# test_plot = ct_boundary.merge(test_mask, on='town', how='left')
# ct_towns.plot(column='Percent_Affordable', cmap='YlOrRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax)
# ax.set_aspect('equal')
# print(ct_df_2017.violent_crime)
ct_df_2017.plot(column='violent_crime', cmap='PuRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax3)
ax3.axis('off')
ax3.set_title('Violent Crimes by\nTown in Connecticut, 2017')

ct_towns17.plot(column='Percent_Affordable', cmap='PuRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax4)
ax4.axis('off')
ax4.set_title('Percent Affordable Housing by\nTown in Connecticut, 2017')
plt.show()


crimes = [df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017]
uconn_df = pd.DataFrame()
n = 0
for data in crimes:
    data['Year'] = 2011 + n
    uconn_df = uconn_df.append(data.iloc[174, :])
    n += 1

# print(uconn_df)

mansfield = pd.DataFrame()
mansfield = df[df['Town'] == 'Mansfield']

uconn_df = uconn_df.merge(mansfield, on='Year', how='left')
# print(uconn_df.head())





fig = plt.figure() # Create matplotlib figure

ax5 = fig.add_subplot(111) # Create matplotlib axes
ax6 = ax5.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

uconn_df.violent_crime.plot(kind='bar', color='lightpink', ax=ax5, width=width, position=1)
uconn_df.Percent_Affordable.plot(kind='bar', color='deepskyblue', ax=ax6, width=width, position=0)

ax5.set_ylabel('Violent Crimes (pink)')
ax6.set_ylabel('Percent Affordable Housing (blue)')
ax5.set_title('Violent Crime at UConn \nvs \nPercent Affordable Housing in Mansfield 2011-2017')

labels = [item.get_text() for item in ax5.get_xticklabels()]

# replace the x-axis tick labels with the values from the 'year' column
ax5.set_xticklabels(uconn_df['Year'])

plt.show()





sum_df = df.groupby('Year').sum()

# reset the index to make 'year' a column again
sum_df = sum_df.reset_index()

# show the resulting dataframe with the sums for each year
# print(sum_df)



connecticut = pd.DataFrame()
n = 0
for data in crimes:
    data['Year'] = 2011 + n
    connecticut = connecticut.append(data.iloc[1, :])
    n += 1


connecticut = connecticut.merge(sum_df, on='Year', how='left')
# print(connecticut.columns)





fig = plt.figure() # Create matplotlib figure

ax7 = fig.add_subplot(111) # Create matplotlib axes
ax8 = ax7.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

connecticut.violent_crime.plot(kind='bar', color='cyan', ax=ax7, width=width, position=1)
connecticut.Total_Assisted_Units.plot(kind='bar', color='darkorchid', ax=ax8, width=width, position=0)

ax7.set_ylabel('Violent Crimes (cyan)')
ax8.set_ylabel('Affordable Housing Units (purple)')
ax7.set_title('Violent Crime \nvs \nTotal Affordable Housing in CT 2011-2017')

labels = [item.get_text() for item in ax7.get_xticklabels()]

# replace the x-axis tick labels with the values from the 'year' column
ax7.set_xticklabels(connecticut['Year'])

plt.show()



## datasets = df affordable housing, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017 for crimes

towns_2011 = df_2011.iloc[2:173]
towns_2012 = df_2012.iloc[2:173]
towns_2013 = df_2013.iloc[2:173]
towns_2014 = df_2014.iloc[2:173]
towns_2015 = df_2015.iloc[2:173]
towns_2016 = df_2016.iloc[2:173]
towns_2017 = df_2017.iloc[2:173]


print('ANOVA FOR NON VIOLENT AND VIOLENT')
print(stats.f_oneway(towns_2011.non_violent_crime, towns_2012.non_violent_crime, towns_2013.non_violent_crime, towns_2014.non_violent_crime, towns_2015.non_violent_crime, towns_2016.non_violent_crime, towns_2017.non_violent_crime))
print(stats.f_oneway(towns_2011.violent_crime, towns_2012.violent_crime, towns_2013.violent_crime, towns_2014.violent_crime, towns_2015.violent_crime, towns_2016.violent_crime, towns_2017.violent_crime))
## high p value, so we do not reject the null hypothesis and assume that the averages are the same
for lst in [towns_2011.violent_crime, towns_2012.violent_crime, towns_2013.violent_crime, towns_2014.violent_crime, towns_2015.violent_crime, towns_2016.violent_crime, towns_2017.violent_crime]:
    print(lst.mean())


gd = df.groupby(["Year", "Town"])["Town", "Total_Assisted_Units"].mean().reset_index()
units_2011 = np.array(gd[gd['Year'] == 2011]['Total_Assisted_Units'])
units_2012 = np.array(gd[gd['Year'] == 2012]['Total_Assisted_Units'])
units_2013 = np.array(gd[gd['Year'] == 2013]['Total_Assisted_Units'])
units_2014 = np.array(gd[gd['Year'] == 2014]['Total_Assisted_Units'])
units_2015 = np.array(gd[gd['Year'] == 2015]['Total_Assisted_Units'])
# print(units_2015)
units_2016 = np.array(gd[gd['Year'] == 2016]['Total_Assisted_Units'])
units_2017 = np.array(gd[gd['Year'] == 2017]['Total_Assisted_Units'])
units_2022 = np.array(gd[gd['Year'] == 2022]['Total_Assisted_Units'])


print('ANOVA FOR UNITS OVER YEARS')
# perform one-way ANOVA for each year
print(stats.f_oneway(units_2011, units_2012, units_2013, units_2014, units_2015, units_2016, units_2017))
    
    
for lst in [units_2011, units_2012, units_2013, units_2014, units_2015, units_2016, units_2017, units_2022]:
    print(lst.mean())
print('UNITS 2011 vs 2017') 
print(stats.ttest_ind(units_2011, units_2017, alternative='greater'))
print(stats.ttest_ind(units_2011, units_2017, alternative='less'))
print(stats.ttest_ind(units_2011, units_2017, alternative='two-sided'))

## There is not enough information to reject the null hypothesis that the means are equal

print('UNITS 2011 2022')
print(stats.ttest_ind(units_2011, units_2022, alternative='less'))
## can visually see it increasing but can't statistically prove it at a high confidence level

## do some tests on crime to see if there is statistically prove-able differences

print('\n\n\n\n')
print('VIOLENT CRIME 2011 2017')
print(stats.ttest_ind(towns_2011.violent_crime, towns_2017.violent_crime, alternative='greater'))

df = df.rename(columns={'Town': 'town'})
print(towns_2011.head())
print(df.columns)
print(towns_2011.columns)
merged = pd.merge(df, towns_2017, how='left', on=["town", "Year"])

print(merged.head())

towns_2011['town'] = towns_2011['town'].str.split('_').str[:-1].str.join('_').str.replace('_', ' ')
towns_2012['town'] = towns_2012['town'].str.split('_').str[:-1].str.join('_').str.replace('_', ' ')
towns_2013['town'] = towns_2013['town'].str.split('_').str[:-1].str.join('_').str.replace('_', ' ')
towns_2014['town'] = towns_2014['town'].str.replace('_', ' ')
towns_2015['town'] = towns_2015['town'].str.replace('_', ' ')
towns_2016['town'] = towns_2016['town'].str.replace('_', ' ')
towns_2017['town'] = towns_2017['town'].str.replace('_', ' ')
print(towns_2011.town)

concatenated_towns = pd.concat([towns_2011, towns_2012, towns_2013, towns_2014, towns_2015, towns_2016, towns_2017], axis=0, ignore_index=True)

merged = pd.merge(df, concatenated_towns, left_on=["town", "Year"], right_on=["town", "Year"])
print(merged.iloc[673:680])
from matplotlib.colors import ListedColormap

fig, ax9 = plt.subplots()

# Create the scatter plot
scatter = ax9.scatter(merged['Total_Assisted_Units'], merged['violent_crime'], c=merged['Year'], cmap='Set2', alpha=0.6)

# Add a color bar to show the year values
# colorbar = plt.colorbar(scatter)
# colorbar.set_label('Year')
handles, labels = scatter.legend_elements(prop='colors')
legend = ax9.legend(handles, labels, loc='best', title='Year')

# Set the axis labels
ax9.set_ylabel('Violent Crime')
ax9.set_xlabel('Total Affordable Units')
ax9.set_title('Total Affordable Housing Units vs Violent Crime')

# Show the plot
plt.show()

fig, ax10 = plt.subplots()

# Create the scatter plot
scatter = ax10.scatter(merged['Total_Assisted_Units'], merged['violent_crime'], c=merged['Year'], cmap='Set2', alpha=0.6)

# Add a color bar to show the year values
# colorbar = plt.colorbar(scatter)
# colorbar.set_label('Year')
handles, labels = scatter.legend_elements(prop='colors')
legend = ax10.legend(handles, labels, loc='best', title='Year')

# Set the axis labels
ax10.set_ylabel('Violent Crime')
ax10.set_xlabel('Total Affordable Units')
ax10.set_title('Total Affordable Housing Units vs Violent Crime')

ax10.set_xlim([0, 7000])
ax10.set_ylim([0, 1500])
# Show the plot
plt.show()

merged.to_csv('everything_merged.csv', encoding='utf-8', index=False)


slope, intercept, r_value, p_value, std_err = stats.linregress(merged.Total_Assisted_Units, merged.violent_crime)
print("Slope:", slope)
print("Intercept:", intercept)
print("R value:", r_value)
print("P value:", p_value)
print("Standard error:", std_err)





fig, ax11 = plt.subplots()

# Create the scatter plot
scatter = ax11.scatter(merged['Total_Assisted_Units'], merged['violent_crime'], c=merged['Year'], cmap='Set2', alpha=0.6)

# Add a color bar to show the year values
# colorbar = plt.colorbar(scatter)
# colorbar.set_label('Year')
handles, labels = scatter.legend_elements(prop='colors')
legend = ax11.legend(handles, labels, loc='best', title='Year')

# Set the axis labels
ax11.set_ylabel('Violent Crime')
ax11.set_xlabel('Total Affordable Units')
ax11.set_title('Total Affordable Housing Units vs Violent Crime')

ax11.set_xlim([0, 1000])
ax11.set_ylim([0, 200])
# Show the plot
plt.show()


















