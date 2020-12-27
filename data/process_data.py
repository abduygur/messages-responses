import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Load the messages and categories data

        Parameters
        ----------
        messages_filepath: Messages Data Filepath
        categories_filepath: Categories Data Filepath

        Returns
        -------
        df: Dataframe which is merged messages and categories

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    return df

def clean_data(df):
    """
        Clean the dataframe:
        1- Split the categories into seperate category columns
        2- Convert the category values to just numbers 0 or 1
        3- Replace categories column in df with new category columns
        4- Remove duplicates

        Parameters
        ----------
        df: Merged Messages and Categories

        Returns
        -------
        df: Cleaned Dataframe

    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,]

    category_colnames = [cat.split('-')[0] for cat in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    categories.drop('child_alone', axis=1, inplace=True)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df = df[df.duplicated() == False]
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filename):
    """
        Save the clean dataset into sqlite database

        Parameters
        ----------
        df: Cleaned Dataframe
        database_filename: Sqlite database filepath

        Returns
        -------
        None

    """
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('InsertTableName', engine, index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()