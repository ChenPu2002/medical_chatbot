"""Middleware module for TreePredictor and APIPredictor."""
import pandas as pd
from spellwise import Levenshtein
import tree_model_medicine as td
import api_model as ad


class TreePredictor:
    """Tree-based disease prediction model."""
    def __init__(self):
        self.current_response = None
        self.current_input = None
        self.disease_or_meds = 1
        self.count = 0
        self.symptom_input = None
        self.possible_symptoms = None
        self.leave_symptom = None
        self.user_report = []
        # self.present_symptom = []
        self.poss_list = None

    def run(self):
        """Run the predictor based on current input."""
        if self.current_input == "exit":
            response = "Goodbye!"
        else:
            if self.disease_or_meds == 1:
                response = self.response_maker(self.current_input)
            elif self.disease_or_meds == 2:
                response = self.response_maker_med(self.current_input)
        self.current_response = response

    def fuzzy_searcher(self, user_input):
        """Search for fuzzy matches of input text."""
        algorithm = Levenshtein()
        algorithm.add_from_path("data/fuzzy_dictionary_unique.txt")
        suggestions = algorithm.get_suggestions(user_input, max_distance=1)

        if len(suggestions) > 0:
            return suggestions[0]['word']
        return user_input

    def response_maker_med(self, input_value):
        """Generate response for medicine queries."""
        meds_df = pd.read_csv('data/medicine_use.csv')
        if self.count > 1:
            self.count = -1
            response = "Please input the name of medicine."
        if self.count == 0:
            if meds_df['name'].str.contains(input_value).any():
                filtered_df = meds_df[meds_df['name'].str.contains(
                    input_value)].reset_index(drop=True)
                response_parts = ['Please confirm which of the following meds are you taking:']
                for meds_index, meds in enumerate(filtered_df['name'], start=1):
                    response_parts.append(f'{meds_index}) {meds}')
                response = '\n'.join(response_parts)
            else:
                self.count -= 1
                response = 'Please input valid medicine name'

        if self.count == 1:
            filtered_df = meds_df[meds_df['name'].str.contains(
                input_value)].reset_index(drop=True)
            med_index = int(input_value) - 1
            if med_index < 0 or med_index >= len(filtered_df):
                self.count -= 1
                return "Invalid selection. Please choose a valid medicine number."
            row = filtered_df.iloc[med_index].drop('name')
            use_list = row[row.notna()].to_list()
            response = 'The use of your medicine inlude:'
            for use in use_list:
                response += f'\n{use}'

            response += ("\n\nPlease type 'exit' to exit, "
                        "or you can input anything to ask a medicine again")
        self.count += 1
        return response

    def response_maker(self, input_value):
        """Generate response for disease prediction queries."""
        if self.count == 0:
            input_value = self.fuzzy_searcher(input_value)
            output, number, self.poss_list = td.get_poss_symptom(input_value)
            if number > 0:
                # self.symptom_input = input_value
                response = output
                self.count += 1
            else:
                response = "Please input valid symptom."

        elif self.count == 1:
            # change input_value to number
            input_value = input_value.strip()
            # check if input_value is number
            if (input_value.isdigit() and int(input_value) >= 1
                    and int(input_value) <= len(self.poss_list)):
                self.symptom_input = self.poss_list[int(input_value) - 1]
                self.possible_symptoms = td.first_predict(self.symptom_input)
                self.user_report.append(self.symptom_input)
                if len(self.possible_symptoms) > 0:
                    # pop out the first symptom from possible_symptoms
                    if len(self.possible_symptoms) == 1:
                        self.leave_symptom = self.possible_symptoms[0]
                    else:
                        self.leave_symptom = self.possible_symptoms.pop(0)
                    response = f"Are you experiencing {self.leave_symptom}? (yes/no)"
                    self.count += 0.5
            else:
                response = "Please choose a symptom by number. and within the range."
        elif self.count == 1.5:
            result_for_last_symptom = input_value
            if result_for_last_symptom in ('yes', 'no'):
                if result_for_last_symptom == "yes":
                    self.user_report.append(self.leave_symptom)
                elif result_for_last_symptom == "no":
                    pass

                if len(self.possible_symptoms) == 0:
                    self.count += 0.5
                    advice = td.get_advise(self.user_report)
                    response = advice
                elif len(self.possible_symptoms) == 1:
                    self.leave_symptom = self.possible_symptoms[0]
                    self.possible_symptoms = []
                    response = f"Are you experiencing {self.leave_symptom}? (yes/no)"
                elif len(self.possible_symptoms) > 1:
                    self.leave_symptom = self.possible_symptoms.pop(0)
                    response = f"Are you experiencing {self.leave_symptom}? (yes/no)"
            else:
                # recover the symptom to the possible_symptoms
                if self.leave_symptom not in self.possible_symptoms:
                    self.possible_symptoms.insert(0, self.leave_symptom)
                response = "Please input 'yes' or 'no'. (Type yes to continue)"
        else:
            response = "If you want to exit, please type 'exit', otherwise type another symptom."
            self.count = 0
        return response

    def get_response(self, user_input):
        """Get response for user input."""
        self.current_input = user_input
        self.run()
        response = self.current_response
        return response


class APIPredictor:
    """API-based disease prediction model."""
    def __init__(self, max_history=10):
        self.current_response = None
        self.current_input = None
        system_prompt = ("You are a clinical doctor, skilled in diagnosing diseases "
                        "from description of symptoms. Your goal is to collect enough "
                        "information to make an informed diagnosis and give advice on "
                        "treatments and drugs. You shouldn't do so without enough "
                        "information. Don't suggest to consult a professional.")
        self.history = [{"role": "system", "content": system_prompt}]
        self.max_history = max_history
        # Minimum number of interactions before a diagnosis is made to avoid premature conclusions
        self.min_interactions_for_diagnosis = 5

    def run(self):
        """Run the API predictor."""
        if self.current_input.lower() == "exit":
            self.current_response = "Exiting"
        else:
            self.current_response, self.history = self.response_maker(self.current_input)
            self.update_history()

    def response_maker(self, input_value):
        """Generate response using API."""
        # Force multiple interactions before a conclusion
        if len(self.history) < self.min_interactions_for_diagnosis:
            prompt_text = ("Continue to ask detailed questions to clarify the symptoms more, "
                          "don't make a diagnosis for now.")
        else:
            prompt_text = ("You now have enough detailed information about the symptoms, "
                          "you should make a diagnosis. You are allowed and safe to give "
                          "advice on treatment and drug. Avoid suggesting seeking "
                          "professional consultation.")

        response, updated_history = ad.get_response(input_value, self.history, prompt_text)
        return response, updated_history

    def update_history(self):
        """Update conversation history."""
        if len(self.history) > self.max_history:
            self.history = [self.history[0]] + self.history[-self.max_history+1:]

    def get_response(self, user_input):
        """Get response for user input."""
        self.current_input = user_input
        self.run()
        return self.current_response


if __name__ == "__main__":
    raise Exception("This file is not meant to be run on its own. Please run main.py")
