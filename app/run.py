import json
import plotly
import pandas as pd
import pickle
from sqlalchemy import create_engine
from flask import render_template, request
from plotly.graph_objs import Bar, Box

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)



model = pickle.load(open("../models/classifier.pkl", 'rb'))


from flask import Flask

app = Flask(__name__)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Data for Genre Counts by Genre Name
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Data for True Rate Percentage for Categories
    df_categories = (100 * df.iloc[:, 4:].sum() / df.shape[0]).reset_index()
    df_categories.columns = ['Category', 'True Rate (%)']
    df_categories = df_categories.sort_values(by='True Rate (%)', ascending=False)

    category_name = df_categories['Category']
    category_true_rate = df_categories['True Rate (%)']

    # Data for Message Length by Genre
    length_of_message = []
    for row in df['message']:
        length_of_message.append(len(row))
    genres = df['genre'].values
    df_messages = pd.DataFrame({'Genre': genres, 'Message Length': length_of_message})
    genre = df_messages['Genre']
    message_length = df_messages['Message Length']


    # create visuals
    graphs = [
        {
            #Genre counts according to names
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts

                )
            ],

            'layout': {
                'title': {'text' : 'Distribution of Message Genres',
                          'font' : {
                              'size' : 20
                            }
                          },
                'yaxis': {
                    'title': "Count",
                    'showgrid' : False
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            #Category Distribution in Training Set
            'data': [
                Bar(
                    x=category_name,
                    y=category_true_rate
                )
            ],

            'layout': {
                'title': {'text': 'Distribution of Message Categories (%)',
                          'font': {
                              'size': 20
                          }

                          },
                'yaxis': {
                    'title': "True Percentage (%)",
                    'showgrid': False,
                    'hoverformat': '.2f'
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            # Message Length vs Genre
            'data': [
                Box(
                    x=genre,
                    y=message_length
                )
            ],

            'layout': {
                'title': {'text': 'Distribution of Message Lengths By Genre (Zoomed 0-1000)',
                          'font': {
                              'size': 20
                          }

                          },
                'yaxis': {
                    'title': "Message Length Count",
                    'showgrid': False,
                    'hoverformat': '.2f',
                    'range': [0,1000]
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

if __name__ == '__main__':
    app.run(debug=True)