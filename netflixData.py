import pandas as pd
netflix_data = pd.read_csv('netflix_titles.csv')

#Question 1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

netflix_data = netflix_data.dropna(subset=['release_year'])
netflix_data['release_year'] = netflix_data['release_year'].astype(int)
content_type_count = netflix_data.groupby(['release_year', 'type']).size().unstack().fillna(0)
content_type_count = content_type_count[content_type_count.index >= 2007]
content_type_count.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Trend of Movies vs. TV Shows on Netflix by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.legend(title='Type')
plt.grid(True)
plt.show()

# Question 2
from collections import Counter
import itertools
all_genres = list(itertools.chain(*netflix_data['listed_in'].str.split(', ')))
genre_counts = Counter(all_genres)
top_genres = pd.DataFrame(genre_counts.most_common(10), columns=['Genre', 'Count'])
sns.barplot(x='Count', y='Genre', data=top_genres, palette='viridis')
plt.title('Top 10 Most Popular Genres on Netflix')
plt.show()

# Question 3
top_countries = netflix_data['country'].value_counts().head(10)

sns.barplot(y=top_countries.index, x=top_countries.values, palette='magma')
plt.title('Top 10 Countries Contributing to Netflix Content')
plt.xlabel('Number of Titles')
plt.show()
