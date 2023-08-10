import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

# Takes in a dataset and sample size and returns a stratified sample
# Takes similar number of sentences from each article
def strat_samp(data, sample_size):
    t_size = sample_size / len(data)
    data = data[data['sentence_count'] > 1]

    train_data, test_data = train_test_split(data, test_size=t_size, stratify=data['article_id'], random_state=42)
    return test_data

# List of datasets
df_list = [nyt_2019_op_500_data, huffpost_500_data, nyt_2019_500_data, bbc_2019_500_data,
          reuters_500_data, foxnews_2019_500_data, breitbart_500_data, foxnews_2019_op_500_data]
df_list = [strat_samp(df, 4000) for df in df_list]
left_data = pd.concat([df_list[1], df_list[2]])
center_data = pd.concat([df_list[3], df_list[4]])
right_data = pd.concat([df_list[5], df_list[6]])

# Plots left_data, center_data, and right_data subjectivity box plot distributions
def group_boxplots():

  # Create an example plot and Axes object
  fig, ax = plt.subplots()

  column = 'subjectivity'
  # Plot multiple boxplots in one graph
  boxplot_data = [left_data[column], center_data[column], strat_samp(right_data, 18000)[column]]
  labels = ['Left Leaning', 'Center', 'Right Leaning']
  ax.boxplot(boxplot_data, boxprops=dict(linewidth=2.5), labels=labels)

  # Set labels and title
  ax.set_xlabel('Media Bias', fontsize=16)
  ax.set_xticklabels(fontsize=16, labels=labels)
  ax.set_ylabel(column + ' score', fontsize=16)
  ax.tick_params(axis='y', labelsize=16)
  ax.set_title('Subjectivity Distribution of News Sentences', fontsize=20)
  for i, data in enumerate(boxplot_data):
    mean_val = round(np.mean(data), 2)
    median_val = round(np.median(data), 2)
    std_val = round(np.std(data), 2)
    ax.text(i + 1, np.max(data) + 0.2, f'Mean: {mean_val}\nMedian: {median_val}\nStd: {std_val}',
            ha='center', va='bottom', fontsize=12)


  # Display the plot
  plt.show()
