import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split

import _pickle as cPickle
from languages import languages, load_keywords, strip_non_keywords

dataset_pkl = 'dataset.pkl'
tfidf_pkl = 'tfidf.pkl'
nmf_pkl = 'nmf.pkl'
train_pkl = 'train.pkl'
test_pkl = 'test.pkl'
classifier_pkl = 'classifier.pkl'


def store_pickle(file_name, obj):
    with open(file_name, 'wb') as fd:
        cPickle.dump(obj, fd)


def load_pickle(file_name):
    with open(file_name, 'rb') as fd:
        obj = cPickle.load(fd)
        return obj


def load_dataset(dataset_dir):
    print('Loading dataset...')
    df = pd.DataFrame({'Source': [], 'Label': []})
    if os.path.exists(dataset_pkl):
        df = load_pickle(dataset_pkl)
    else:
        keywords = load_keywords()
        language_dir = os.listdir(dataset_dir)
        for lang in language_dir:
            if lang not in languages:
                continue
            task_file = os.listdir(dataset_dir + lang)
            print('Loading {} files...'.format(lang))
            temp_df = pd.DataFrame({'Source': [], 'Label': []})
            for task in task_file:
                with open(dataset_dir + lang + '/' + task, 'r') as file:
                    data = file.read()
                    data = strip_non_keywords(data, keywords, lang)
                    temp_df = pd.concat([temp_df, pd.DataFrame(
                        {'Source': [data], 'Label': [lang]})], ignore_index=True)
            df = pd.concat([df, temp_df], ignore_index=True)
        store_pickle(dataset_pkl, df)

    print('Loaded a total of {} source files'.format(len(df['Label'])))

    return df


def split_dataset(df):
    print('Splitting dataset into train and test sets...')
    train, test = train_test_split(df, test_size=0.2, shuffle=False)

    return train, test


def reduce_dimension(train_df, test_df):
    if os.path.exists(train_pkl) and os.path.exists(test_pkl):
        print('Loading truncated matrix...')
        x_train = load_pickle(train_pkl)
        y_test = load_pickle(test_pkl)
    else:
        print('Generating sparse matrix...')
        if os.path.exists(tfidf_pkl):
            tfidf = load_pickle(tfidf_pkl)
        else:
            tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(
                7, 7), sublinear_tf=True, token_pattern='.*', min_df=10, max_df=0.99, lowercase=False)
            tfidf.fit(train_df['Source'], train_df['Label'])
            store_pickle(tfidf_pkl, tfidf)

        x_train_source = tfidf.transform(train_df['Source'])
        x_train_label = tfidf.transform(train_df['Label'])

        y_test_source = tfidf.transform(test_df['Source'])
        y_test_label = tfidf.transform(test_df['Label'])

        print('Reducing matrix dimension...')
        if os.path.exists(nmf_pkl):
            model = load_pickle(nmf_pkl)
        else:
            model = NMF(n_components=200, init='nndsvda', max_iter=100)
            model.fit(x_train_source, x_train_label)
            store_pickle(nmf_pkl, model)

        x_train = model.transform(x_train_source)
        y_test = model.transform(y_test_source)

        store_pickle(train_pkl, x_train)
        store_pickle(test_pkl, y_test)
    return x_train, y_test


def train_classifier(x_train, labels):
    if os.path.exists(classifier_pkl):
        print('Loading classifier...')
        clf = load_pickle(classifier_pkl)
    else:
        print('Setting up classifier...')
        clf = RandomForestClassifier(n_estimators=128, min_samples_split=5, class_weight='balanced')
        clf = clf.fit(x_train, labels)
        store_pickle(classifier_pkl, clf)
    return clf


def benchmark_classifier(cls, y_test, test_df, scoring):
    print('Calculating classifier performance...')
    for scr in scoring:
        print(scr, ':', str(np.mean(cross_val_score(cls, y_test, test_df['Label'], cv=10, scoring=scr))))


def train(dataset_dir):
    df = load_dataset(dataset_dir)

    train_df, test_df = split_dataset(df)

    x_train, y_test = reduce_dimension(train_df, test_df)

    clf = train_classifier(x_train, train_df['Label'])

    benchmark_classifier(clf, y_test, test_df, ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])


def predict(file_name):
    if os.path.exists(tfidf_pkl) and os.path.exists(nmf_pkl) and os.path.exists(classifier_pkl):
        tfidf = load_pickle(tfidf_pkl)
        model = load_pickle(nmf_pkl)
        clf = load_pickle(classifier_pkl)
    else:
        print('Could not find pickle files. Use \'- -train\' flag first')

    with open(file_name, 'r') as file:
        data = file.read()
        pred_df = pd.DataFrame({'Source': [data], 'Label': ['?']})

        y_test_source = tfidf.transform(pred_df['Source'])
        y_test_label = tfidf.transform(pred_df['Label'])

        y_test = model.transform(y_test_source)

        test_pred = clf.predict(y_test)
        print(file_name, ':', test_pred[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        help='Train a new model using the provided dataset. The argument must be a directory \
                        containing one directory for each of the supported languages(as defined in languages_list).')
    parser.add_argument('--predict', type=str,
                        help='Predict the lang of the provided file. File extension is not used to \
                        define the lang of the source file. If used with the \'--train\' flag then a new \
                        model will be first trained before predicting the lang of the file.')
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)
    if args.predict is not None:
        predict(args.predict)


if __name__ == '__main__':
    main()
