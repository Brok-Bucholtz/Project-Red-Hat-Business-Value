from copy import copy

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import sys


def _people_ids_to_ints(people_ids):
    people_ids = people_ids.apply(lambda x: x.split('_')[1])
    return pd.to_numeric(people_ids).astype(int)


def _actions_to_ints(actions):
    int_actions = copy(actions)
    int_actions = int_actions.fillna('type 0')
    int_actions = int_actions.apply(lambda x: x.split(' ')[1])
    return pd.to_numeric(int_actions).astype(int)


def _preprocess_actions(data):
    data = data.drop(['date', 'activity_id'], axis=1)
    data['people_id'] = _people_ids_to_ints(data['people_id'])
    
    for column in data.columns:
        if column != 'people_id':
            data[column] = _actions_to_ints(data[column])
    return data


def _preprocess_people(data):
    string_columns = [
        'char_1', 'group_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9']
    data = data.drop(['date'], axis=1)
    data['people_id'] = _people_ids_to_ints(data['people_id'])

    for column in string_columns:
        data[column] = _actions_to_ints(data[column])
    return data


def _run_final_model(clf):
    test_actions = pd.read_csv('./input/act_test.csv')
    people = pd.read_csv('./input/people.csv')

    people = _preprocess_people(people)
    test_ids = test_actions['activity_id']
    test_actions = _preprocess_actions(test_actions)
    test_actions = test_actions.merge(people, how='left', on='people_id')

    output = pd.DataFrame({'activity_id': test_ids, 'outcome': clf.predict(test_actions)})
    output.head()
    output.to_csv('redhat.csv', index=False)


def _get_features_labels():
    train_actions = pd.read_csv('./input/act_train.csv')
    people = _preprocess_people(pd.read_csv(
        './input/people.csv',
        dtype={
            'char_10': np.int, 'char_11': np.int, 'char_12': np.int, 'char_13': np.int, 'char_14': np.int,
            'char_15': np.int, 'char_16': np.int, 'char_17': np.int, 'char_18': np.int, 'char_19': np.int,
            'char_20': np.int, 'char_21': np.int, 'char_22': np.int, 'char_23': np.int, 'char_24': np.int,
            'char_25': np.int, 'char_26': np.int, 'char_27': np.int, 'char_28': np.int, 'char_29': np.int,
            'char_30': np.int, 'char_31': np.int, 'char_32': np.int, 'char_33': np.int, 'char_34': np.int,
            'char_35': np.int, 'char_36': np.int, 'char_37': np.int, 'char_38': np.int}))

    labels = train_actions['outcome']
    train_actions = train_actions.drop(['outcome'], axis=1)
    train_actions = _preprocess_actions(train_actions)
    features = train_actions.merge(people, how='left', on='people_id')

    return features, labels


def run():
    TEST_SIZE = 0.20
    features, labels = _get_features_labels()
    clf = RandomForestClassifier()

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)
        clf.fit(x_train, y_train)
        score = roc_auc_score(y_test, clf.predict(x_test))
        print('Area under ROC {}'.format(score))
    else:
        clf.fit(features, labels)
        _run_final_model(clf)


if __name__ == '__main__':
    run()
