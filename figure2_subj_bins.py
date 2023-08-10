import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
from sklearn.model_selection import train_test_split

def strat_samp(data, sample_size):
    t_size = sample_size / len(data)
    data = data[data['sentence_count'] > 1]
    train_data, test_data = train_test_split(data, test_size=t_size, stratify=data['article_id'], random_state=42)
    return test_data

# List of datasets
df_list = [nyt_2019_op_500_data, huffpost_500_data, nyt_2019_500_data, bbc_2019_500_data,
          reuters_500_data, foxnews_2019_500_data, breitbart_500_data, foxnews_2019_op_500_data]
df_list = [strat_samp(df, 4000) for df in df_list]


# Set emotion and emotion threshold
emot = 'anger'
emotion_range = 0.6

# Filter the data
left_data = pd.concat([df_list[1], df_list[2]])
center_data = pd.concat([df_list[3], df_list[4]])
right_data = pd.concat([df_list[5], df_list[6]])


# Create the figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))


def plot(df, ax, col, title):
  bins = np.linspace(0, 1, 11)
  # Filter the data by the emotion level and the corresponding sentiment intensity
  filtered_data = df[(df[emot] > emotion_range) & (df['sentiment_intensity'] < -emotion_range)]

  # Calculate the percentage of data in each bin for the filtered data
  filtered_data['subjectivity_bin'] = pd.cut(filtered_data['subjectivity'], bins=bins, include_lowest=True)
  filtered_distribution = filtered_data['subjectivity_bin'].value_counts(normalize=True) * 100

  # Convert the bin categories to intervals and sort
  filtered_distribution.index = filtered_distribution.index.categories
  filtered_distribution = filtered_distribution.sort_index()

  # Plot the histogram for the filtered data
  axs[ax].bar(filtered_distribution.index.astype(str), filtered_distribution.values, color=col, alpha=0.7)
  axs[ax].set_title(title, fontsize=18)
  axs[ax].set_xlabel('Subjectivity Score Bin', fontsize=16)
  axs[ax].set_ylabel('Percentage (%)', fontsize=16)
  axs[ax].set_ylim([0, 35])
  axs[ax].tick_params(axis='x', rotation=45, labelsize=14)
  axs[ax].annotate(f'Total # of Sentences: {len(df)} \n % Filtered: {round(len(filtered_data) / len(df) * 100, 2)}', xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=14, ha='right', va='top')

  # Annotate each bar with its value
  for i, count in enumerate(filtered_distribution.values):
      axs[ax].annotate(f'{round(count, 2)}', xy=(i, count), xytext=(0, 3), textcoords='offset points',
                  ha='center', va='bottom', fontsize=10)

plot(left_data, 0, 'red', 'Left Leaning Sources (HuffPost & NYT)')
plot(center_data, 1, 'blue', 'Center Sources (Reuters & BBC)')
plot(right_data, 2, 'green', 'Right Sources (Breitbart & Fox)')

# Show the plot
plt.suptitle(f'Subjectivity Distribution of Sentences w/ {emot.capitalize()} >= {emotion_range:.1f}, Sentiment <= {-emotion_range:.1f}', fontsize=20)
plt.tight_layout()
plt.show()

