import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def strat_samp(data, sample_size):
    t_size = sample_size / len(data)
    data = data[data['sentence_count'] > 1]
    train_data, test_data = train_test_split(data, test_size=t_size, stratify=data['article_id'], random_state=42)
    return test_data

subj_thres = 0.8
sent_thres = 0.6

# List of datasets
df_list = [nyt_2019_op_500_data, huffpost_500_data, nyt_2019_500_data, bbc_2019_500_data,
          reuters_500_data, foxnews_2019_500_data, breitbart_500_data, foxnews_2019_op_500_data]
df_list = [strat_samp(df, 4000) for df in df_list]


labels = ['NYT Opinion (L)','HuffPost (L)', 'NYT (L)', 'BBC (C)', 'Reuters (C)', 'Fox (R)', 'Breitbart (R)', 'Fox Opinion (R)']

# Function to calculate the percentage of data that fits sentiment intensity and subjectivity threshold
def calculate_percentage(df):
    total = len(df)
    count = len(df[(df['sentiment_magnitude'] > sent_thres) & (df['subjectivity'] > subj_thres)])
    return (count / total) * 100

# Calculate percentages for all eight datasets
perc = [calculate_percentage(df) for df in df_list]

# Function to calculate the percentages of dominant emotion in data that fits sentiment intensity and subjectivity threshold
def calculate_emotion_percentages(df):
    filtered_data = df[(df['sentiment_magnitude'] > sent_thres) & (df['subjectivity'] > subj_thres)]
    total = len(filtered_data)

    # Identify the dominant emotion for each row
    emotions = ['anger', 'fear', 'disgust', 'joy', 'sadness', 'surprise']
    dominant_emotions = filtered_data[emotions].idxmax(axis=1)

    # Calculate the percentages of each emotion
    anger_percentage = (dominant_emotions == 'anger').sum() / total * 100
    fear_percentage = (dominant_emotions == 'fear').sum() / total * 100
    disgust_percentage = (dominant_emotions == 'disgust').sum() / total * 100

    # testing other emotions
    joy_percentage = (dominant_emotions == 'joy').sum() / total * 100
    sadness_percentage = (dominant_emotions == 'sadness').sum() / total * 100
    surprise_percentage = (dominant_emotions == 'surprise').sum() / total * 100

    print(anger_percentage, fear_percentage, disgust_percentage,
          joy_percentage, sadness_percentage, surprise_percentage)

    return anger_percentage, fear_percentage, disgust_percentage, joy_percentage, sadness_percentage, surprise_percentage

# Calculate emotion percentages for all eight datasets
anger_percs = []
fear_percs = []
disgust_percs = []
joy_percs = []
sad_percs = []
surp_percs = []

for i, df in enumerate(df_list):
    anger_perc, fear_perc, disgust_perc, joy_perc, sad_perc, surp_perc = calculate_emotion_percentages(df)
    norm_anger_perc = anger_perc / 100 * perc[i]
    norm_fear_perc = fear_perc / 100 * perc[i]
    norm_disgust_perc = disgust_perc / 100 * perc[i]
    norm_joy_perc = joy_perc / 100 * perc[i]
    norm_sad_perc = sad_perc / 100 * perc[i]
    norm_surp_perc = surp_perc / 100 * perc[i]
    anger_percs.append(norm_anger_perc)
    fear_percs.append(norm_fear_perc)
    disgust_percs.append(norm_disgust_perc)
    joy_percs.append(norm_anger_perc)
    sad_percs.append(norm_fear_perc)
    surp_percs.append(norm_disgust_perc)

# Create stacked bar graph
plt.figure(figsize=(12, 6))
barWidth = 0.6
bars1 = plt.bar(labels, anger_percs, color='#b5ffb9', edgecolor='grey', width=barWidth, label='Anger')
bars2 = plt.bar(labels, fear_percs, bottom=np.array(anger_percs), color='#f9bc86', edgecolor='grey', width=barWidth, label='Fear')
bars3 = plt.bar(labels, disgust_percs, bottom=np.array(anger_percs) + np.array(fear_percs), color='#a3acff', edgecolor='grey', width=barWidth, label='Disgust')
bars4 = plt.bar(labels, sad_percs, bottom=np.array(anger_percs) + np.array(fear_percs) + np.array(disgust_percs), color='#F08080', edgecolor='grey', width=barWidth, label='Sadness')
bars5 = plt.bar(labels, joy_percs, bottom=np.array(anger_percs) + np.array(fear_percs) + np.array(disgust_percs) + np.array(sad_percs), color='#F9E79F', edgecolor='grey', width=barWidth, label='Joy')
bars6 = plt.bar(labels, surp_percs, bottom=np.array(anger_percs) + np.array(fear_percs) + np.array(disgust_percs) + np.array(sad_percs) + np.array(joy_percs), color='#D2B4DE', edgecolor='grey', width=barWidth, label='Surprise')

plt.xticks(fontsize=20, rotation=20)
plt.ylabel('Percentage', fontsize=25)
plt.title('Comparing Percentage of Sentences w/ High Sentiment ( > ' + str(sent_thres) + ') & High Subjectivity ( > ' + str(subj_thres) + ') & Dominant Emotion')
plt.grid(True, axis='y')
# Get the legend object
legend = plt.legend(prop={'size': 20})

# Set the hue label size
for label in legend.get_texts():
    label.set_fontsize(20)

plt.show()
