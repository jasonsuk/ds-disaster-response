# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Load dataset


def load_data(messages_filepath, categories_filepath):
    ''' Load the two disaster response datsets and merge them on id
    categories records without messages are of no use therefore
    left join will be performed to message data

    INPUT  : file path to message and category data (in order)
    OUTPUT : a merged dataframe
    '''

    # Load as dataframe
    messages_df = pd.read_csv(messages_filepath, sep=',')
    categories_df = pd.read_csv(categories_filepath, sep=',')

    # Left merge

    try:
        df = messages_df.merge(categories_df, on='id')
        return df

    except:
        print('The two files could not be merged.')


# Clean the merged raw data
def clean_data(df):
    ''' Preparing data by key wranling processes identified
    with the sample data sets. Steps are specified in code blocks

    INPUT  : merged dataframe containing messages and categories raw data
    OUTPUT : a cleaned dataframe
    '''

    # 1. Split 'categories' feature into seperate category columns
    # Check if 'categories' column exists
    if len(df['categories']) > 0:
        # Create a category list by using the first row of 'categories' column
        category_list = df.loc[0, 'categories'].split(';')
        # Removing binary numbers from the category names
        category_list = [category.split('-')[0] for category in category_list]

    # Create a new dataframe of expanded columns of different categories
    categories_expanded = df['categories'].str.split(';', expand=True)
    categories_expanded.columns = category_list

    # 2. Convert category values to just numbers 0 or 1
    # --> i.e. related-0 should become 0 only
    for column in categories_expanded:
        # Set each value to be the last value of the string
        categories_expanded[column] = categories_expanded[column].str.split(
            '-').str[1]
        # Convert column from string to numeric
        categories_expanded[column] = categories_expanded[column].astype(int)

    # Replace integer 2 to 1 for 'related' column
    categories_expanded['related'] = categories_expanded['related'].apply(
        lambda x: 1 if x == 2 else x)

    # 3. Replace 'categories ' column in df with new category columns
    df = pd.concat([df, categories_expanded], axis=1)
    # Drop the 'categories' column that is no longer necessary
    df = df.drop(columns='categories')

    # 4. Remove duplicates (without subsetting) and confirm
    # duplicates were removed
    original_count = df.shape[0]

    df = df.drop_duplicates(keep='first', ignore_index=True)
    print(f'    Dropped {original_count-df.shape[0]} rows due to duplication.')
    # Printing the shape of the final clean dataframe
    print(
        f'    Returning the final data that contains {df.shape[0]} rows x {df.shape[1]} columns.')

    return df


# Save cleaned data to a database
def save_data(df, database_filepath):
    # Use SQLAlchemy library to create a database
    engine = create_engine(f'sqlite:///+{database_filepath}')
    # save the dataframe as 'messages' table in the database
    df.to_sql('messages', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
