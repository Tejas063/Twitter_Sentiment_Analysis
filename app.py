from flask import Flask, render_template, request,redirect,session,url_for

import pickle
import pandas as pd
import snscrape.modules.twitter as sntwitter

def preprocess_text(df, column_name):
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Apply the text preprocessing steps
    df_copy[column_name] = df_copy[column_name] \
        .str.replace(r'(?:@|#|https?:|www\.)\S+', '') \
        .str.replace(r'[^A-Za-z0-9 ]+', '') \
        .str.split() \
        .str.join(' ') \
        .str.lower()
    return df_copy


def load_models():
    # Load the vectoriser.
    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    return vectoriser, LRmodel


def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(text)
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text, pred))

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df


# Loading the models.
vectoriser, LRmodel = load_models()

test_data = []

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('home.html')


@app.route('/custom', methods=['GET','POST'])
def predictmytweet():
    if request.method == 'POST':
        text = request.form['tweet']
        test_data.insert(0,text)
        df = pd.DataFrame({'tweets': test_data})
        preprocessed_df = preprocess_text(df, 'tweets')
        df1 = predict(vectoriser, LRmodel ,preprocessed_df['tweets'])
        prediction = df1['sentiment'].iat[0]
        return render_template('index.html', t=prediction)
    else:
        return render_template('index.html')


@app.route('/topics', methods=['GET', 'POST'])
def topics():
    return render_template('index2.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    topic = request.form['topic']
    return redirect(url_for('dataframe', topic=topic))


@app.route('/dataframe/<topic>', methods=['GET','POST'])
def dataframe(topic):
    # Use snscrape to search Twitter for recent tweets related to the topic
    tweets = []
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(topic + ' lang:en since:2020-01-01 until:2023-04-10').get_items()):
        if i > 9:
            break
        tweets.append([tweet.date, tweet.content, tweet.user.username])

        # Convert the results into a pandas DataFrame
    df = pd.DataFrame(tweets, columns=['Datetime', 'Text', 'Username'])
    text1 = list(df['Text'])
    test_data.clear()
    test_data.extend(text1)
    df1 = predict(vectoriser, LRmodel, test_data)
    sentiment_col = df1['sentiment']
    df = df.join(sentiment_col)
    styled_df = df.style.set_properties(**{'max-width': '100%'})
    styled_html = styled_df.to_html(index=False)
    # Render the DataFrame as an HTML table
    return render_template('dataframe.html',topic=topic,data=styled_html)


if __name__ == '__main__':
    app.run(debug=True)


