import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load dataframe from filepaths - messages and categories
    Input: 
    file path of messages data
    file path of categories data
    output: df - pandas DataFrame, merged dataset from messages and categories
    """
    
    # message dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

    

def clean_data(df):
    '''
    Clean the input pandas DataFrame
    Input: df - pandas DataFrame.
    Output：df - cleaned pandas DataFrame.
    '''
    categories = df["categories"].str.split(";", expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0,:]
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    df['related'] = df['related'].map({0:0, 1:1, 2:1})
    return df


def save_data(df, database_filename):
    """
    Save the cleaned data.
    Input:
    df-pandas dataframe containing cleaned data of messages and categories.
    filename for the output database.
    Output:None.
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('message', engine, index = False)


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