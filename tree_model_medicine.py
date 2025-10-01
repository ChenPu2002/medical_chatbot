"""Tree model for disease prediction based on symptoms."""
import re
import csv
import warnings
import random
from collections import defaultdict
import pandas as pd
import joblib
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

training = pd.read_csv('data/training.csv')

cols = training.columns
cols = cols[:-1]

JOBLIB_FILE = "model/rfc.model"
rf = joblib.load(JOBLIB_FILE)

description_list = {}
precautionDictionary = {}

symptoms_dict = {}

for index, symptom in enumerate(cols):
    symptoms_dict[symptom] = index

def get_description():
    """Load disease descriptions from CSV file."""
    with open('data/symptom_Description.csv', mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def get_precaution_dict():
    """Load precaution dictionary from CSV file."""
    with open('data/symptom_precaution.csv', mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(search_list, user_input):
    """Check if input pattern matches any item in search list."""
    pred_list = []
    user_input = user_input.replace(' ', '_')
    patt = f"{user_input}"
    regexp = re.compile(patt)
    pred_list = [item for item in search_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    return 0, []

chk_dis = ",".join(cols).split(",")


def get_poss_symptom(symptom_input):
    """Get possible symptoms matching user input."""
    conf, cnf_dis = check_pattern(chk_dis, symptom_input)
    output = "searches related to input: \n"
    for num, item in enumerate(cnf_dis):
        num += 1
        output += f"{num}) {item}\n"
    if len(cnf_dis) == 1:
        output += ("Is this the symptom you are experiencing? "
                  "(type 1 to continue):  \n\n if none type 0 to search again.")
    else:
        output += (f"Select the one you meant (1 - {len(cnf_dis)}):  "
                  "\n\n if none type 0 to search again.")
    return output, conf, cnf_dis


def first_predict(symptom_input):
    """Predict possible symptoms based on initial input."""
    symptom_input = symptom_input.strip()
    df = training.groupby(training['prognosis']).mean()
    poss_disease = df[df[symptom_input] > 0.9].index.tolist()
    # seeach all the diseases with value>0.9 symptoms
    poss_symptom = []
    for dis in poss_disease:
        high_value_columns = df.loc[dis, df.loc[dis] > 0.5].index.tolist()
        for item in high_value_columns:
            if item not in poss_symptom:
                poss_symptom.append(item)

    symptom_dict = defaultdict(list)
    for symptom_item in poss_symptom:
        for word in symptom_item.split('_'):
            symptom_dict[word].append(symptom_item)

    # create a set to store unselected symptoms
    unselected_symptoms = set()

    # create a set to store selected symptoms
    selected_symptoms = set()

    # iterate through the symptom_dict
    for word, symptoms in symptom_dict.items():
        if len(symptoms) > 1:  # if there are more than one symptom related to the word
            selected = random.choice(symptoms)  # randomly select one symptom
            # if the selected symptom is not in the selected set
            if selected not in selected_symptoms:
                selected_symptoms.add(selected)  # add the selected symptom to the selected set
                # add the rest of the symptoms to the unselected set
                unselected_symptoms.update(symptoms)
                # remove the selected symptom from the unselected set
                unselected_symptoms.remove(selected)

    # convert the set to list
    unselected_symptoms_list = list(unselected_symptoms)

    # remove items from poss_symptom if items in selected_symptoms
    poss_symptom = [item for item in poss_symptom
                    if item not in unselected_symptoms_list]
    # remove if items in selected_symptoms == symptom_input
    poss_symptom = [item for item in poss_symptom if item != symptom_input]
    if len(poss_symptom) > 8:
        # random select 8 symptoms
        poss_symptom = np.random.choice(poss_symptom, 8, replace=False).tolist()

    # replace '_' with ' ' in poss_symptom
    poss_symptom = [item.replace('_', ' ') for item in poss_symptom]

    return poss_symptom


def get_advise(user_report):
    """Get diagnosis and advice based on user symptoms."""
    # project items in user_report to index by symptoms_dict
    input_vector = np.zeros(len(symptoms_dict))

    for item in user_report:
        # replace ' ' with '_'
        item = item.replace(' ', '_')
        input_vector[[symptoms_dict[item]]] = 1

    second_prediction = rf.predict([input_vector])[0]
    output = ""
    output += "You may have " + second_prediction + "\n"
    output += description_list[second_prediction] + "\n"

    precution_list = precautionDictionary[second_prediction]
    output += "\nTake following measures:\n\n"
    for i, j in enumerate(precution_list):
        if j != "":
            output += str(i+1) + ") " + j + "\n"
    output += "\nType anything to continue."
    return output

get_description()
get_precaution_dict()

if __name__ == "__main__":
    raise Exception("This file is not meant to run")
