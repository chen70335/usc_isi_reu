# Utilizes Google NLP and Huggingface Transformer-based model APIs to derive sentence level text analysis
# scores as a dataset given a dataset of news articles

from transformers import pipeline
from google.cloud import language_v1 # set GOOGLE_APPLICATION_CREDENTIALS=KEY_PATH
import pandas as pd
import spacy

# Turns a list of articles into lists of lists of sentences
def articles_to_sentences(article_list):
    return_list = []
    # nlp_simple = English()
    # nlp_simple.add_pipe('sentencizer')
    nlp_better = spacy.load('en_core_web_lg')
    sent_count = 0
    for article in article_list:
        sentences = []
        for sent in nlp_better(article).sents:
            sent_count += 1
            sentences.append(sent)
        return_list.append(sentences)
    print(sent_count)
    return return_list

# Creates a dataframe with necessary data for analysis
def create_dataframe():
    df = pd.DataFrame({
        'id': [],
        'subjectivity': [],
        'sentiment_intensity': [],
        'sentiment_magnitude': [],
        'anger': [],
        'disgust': [],
        'fear': [],
        'joy': [],
        'neutral': [],
        'sadness': [],
        'surprise': [],
        'article_id': [],
        'sentence_count': [],
        'sentence': []
    })
    return df


# Return a list of apis in the order of Sentiment, subjectivity, and emotion
# Uses Google Natural Language API and Huggingface Pre-trained Transformer-based models
def init_apis():
    # Google Natural Language API
    sentiment = language_v1.LanguageServiceClient()

    # bert base subjectivity classification
    subjectivity = pipeline(task="text-classification",
                            model="cffl/bert-base-styleclassification-subjective-neutral",
                            return_all_scores=True)

    # english roberta base emotion classification
    emotion = pipeline("text-classification",
                       model="j-hartmann/emotion-english-distilroberta-base",
                       return_all_scores=True)
    return [sentiment, subjectivity, emotion]

# Uses Google Natural Language API and Huggingface Pre-trained Transformer-based models
def get_value_api(text, type: str, model):
    text = str(text).replace('�', '')
    if type == "sentiment":
        document = language_v1.types.Document(
            content=text, type_=language_v1.types.Document.Type.PLAIN_TEXT, language="en"
        )
        sentiment = model.analyze_sentiment(
            request={"document": document}
        ).document_sentiment
        return sentiment
    elif type == "subjectivity":
        classification = model(text)
        return classification[0][0]['score']
    # returns a dict where keys are emotions, values are scores
    elif type == "emotion":
        emote_dict = {}
        analysis = model(text)[0]
        for emote in analysis:
            emote_dict[emote['label']] = emote['score']
        return emote_dict

# Takes in a POLUSA dataset of news articles and a title for the output csv file
# Returns a dataset of emotion, subjectivity, and sentiment scores on a per sentence level
def main(data, csv_title):
    output_df = create_dataframe()
    data = articles_to_sentences(data)
    google_senti, bert_subj, roberta_emo = init_apis()
    total_art_left = len(data)
    total_sent_count = 0
    graph_dict = create_dataframe()
    for id, art in enumerate(data):
        total_art_left -= 1
        for sent in art:
            total_sent_count += 1
            try:
                senti = get_value_api(sent, 'sentiment', google_senti)
            except:
                pass
            subje = get_value_api(sent, 'subjectivity', bert_subj)
            emote_dict = get_value_api(sent, 'emotion', roberta_emo)
            graph_dict['id'].append(total_sent_count)
            graph_dict['sentiment_intensity'].append(senti.score)
            graph_dict['sentiment_magnitude'].append(senti.magnitude)
            graph_dict['subjectivity'].append(subje)
            for emote in emote_dict.keys():
                graph_dict[emote].append(emote_dict[emote])
            graph_dict['article_id'].append(id)
            graph_dict['sentence_count'].append(len(art))
            graph_dict['sentence'].append(str(sent))
            print(total_sent_count, " Sentences Complete")
            if total_art_left == len(data) - 1:
                print(graph_dict)
        print(total_art_left, " Articles Remaining")
    for col in graph_dict.keys():
        output_df[col] = graph_dict[col]
    output_df.to_csv(csv_title)


if __name__ == "__main__":
    # Example Curation of WSJ Data
    wsj_2019_df = pd.read_csv('data/wsj_2019_articles.csv')
    wsj_2019_data = list(wsj_2019_df['body'])
    main(wsj_2019_data, 'data/wsj_2019_621.csv')
