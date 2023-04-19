import pandas as pd

df = pd.read_csv('Uniform_Crime_Reporting_System_Arrests_2016.csv')
df.columns = df.columns.str.replace(' ', '_')
print(df.head())

# df.insert(2,'Connecticut',0)
# print(df.columns)

grouped = df.groupby(df['stat_index'].str.split().str[0]).agg('sum') / 3
grouped = grouped.reset_index() # Reset the index to make the 'stat_key' column a regular column
grouped = grouped.drop('stat_index', axis=1) # Drop the existing 'stat_key' column
grouped['stat_index'] = df.groupby(df['stat_index'].str.split().str[0]).first()['stat_index'].values
df_result = grouped

df = df_result.reindex(columns=['stat_index'] + list(df_result.columns[:-1]))

df['stat_index'] = df['stat_index'].str.split().str[0]


df_t = df.set_index('stat_index').T
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
# rename the index to 'key'
df_t.index.name = None
df_t = df_t.reset_index()
df_t = df_t.round(0)
print(df_t.columns)
df_t.iloc[0] = df_t.iloc[0].astype(str)
print(df_t.columns.dtype)
print(df_t.head())

cols_to_convert = df_t.columns[2:]
df_t[cols_to_convert] = df_t[cols_to_convert].astype(float)

violent = ['AgAsslt', 'Murder', 'NgMansl', 'Rape', 'SexOff', 'SmAsslt']
df_t['violent_crime'] = df_t.loc[:, ['AgAsslt', 'Murder', 'NgMansl', 'Rape', 'SexOff', 'SmAsslt']].sum(axis=1)
df_t['non_violent_crime'] = df_t.loc[:, ~df_t.columns.isin(['AgAsslt', 'Murder', 'NgMansl', 'Rape', 'SexOff', 'SmAsslt'])].sum(axis=1)

# print('yes')
# print(df_t.iloc[2])
# df_t.iloc[1] = df_t.iloc[2:173].sum()
# df_t.at[1, 'index'] = 'Connecticut'

df_t.to_csv('2016_crime.csv', encoding='utf-8', index=False)