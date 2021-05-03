import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
import pickle

def load_data(database_filepath):
    """
    Load data from database
    Input: database_filepath including the cleaned data
    Output:
    X: Message (the feature data).
    y: Category of each message (the target).
    category_name: List including the category.
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('message', con=engine)
    
    X = df['message']
    y = df.loc[:, 'related':'direct_report']
    category_name = y.columns.tolist()
    
    return X, y, category_name


def tokenize(text):
    """
    Tokenize text data for modeling process.
    Input:text representing messages
    Output: cleaned and tokenized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Detect URL
    detected_urls = re.findall(url_regex, text) 
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Tokenize text data
    token = word_tokenize(text)
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Proceed cleaning tokens
    clean_token = []
    for tokn in token:
        clean_tokn = lemmatizer.lemmatize(tokn).lower().strip()
        clean_token.append(clean_tokn)
    return clean_token


def build_model():
    """
    Build Machine Learning pipeline using tfidf, Random Forest, and Gridsearch
    Input: None
    Output: cv Results of GridSearch
    """
    # Create pipeline with Classifier
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {
        'clf__estimator__min_samples_split': [2, 5],
        'clf__estimator__max_features': [None, 'log2', 'sqrt'],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [10, 50, None],
    }
    # instantiate a gridsearchcv object with the params defined
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1, cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_name):
    '''
    Evaluate model performance using the test data
    Input: 
    model: model to be evaluated
    X_test: test data (the feature data)
    Y_test: true lables for test data
    category_name: labels for the included categories (N=36)
    Output: accuracy and classfication report of each category
    '''
    # Get results and add them to a dataframe.
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_name)):
        print("Category:", 
              category_name[i],"\n", 
              classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_name[i], 
                                         accuracy_score(Y_test.iloc[:, i].values, 
                                                        Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_name = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_name)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()