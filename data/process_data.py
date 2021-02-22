import sys
import pandas as pd
from sqlalchemy import *


def load_data(messages_filepath, categories_filepath):
    """
    This function load datasets
    input:
        messages and categories datasets
    output:
        Combined data containing messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    This function clean datasets
    input:
        Combined data containing messages and categories
    output:
        Combined clean data containing messages and categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    # drop duplicates
    df.drop_duplicates(inplace=True)
    df = df[df['related'] != 2]
    return df


def save_data(df):
    """
    Save Data to SQLite Database
    """
    #engine = create_engine('sqlite:///' + database_filepath)
    df.to_csv('data/messages_response.csv', index=False)


def main():
    messages_filepath='data/disaster_messages.csv'
    categories_filepath='data/disaster_categories.csv'
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n')
    save_data(df)



if __name__ == '__main__':
    main()