import csv
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

def load_data(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            try:
                text = row[2]  # Assume 'text' is at index 2
                label = row[1]  # Assume 'label' is at index 1
                data.append((text, label))
            except IndexError:
                print(f"Skipping malformed row: {row}")
    return data

bot = ChatBot('MedBot', read_only=True,
              preprocessors=['chatterbot.preprocessors.convert_to_ascii',
                             'chatterbot.preprocessors.unescape_html',
                             'chatterbot.preprocessors.clean_whitespace'],
              logic_adapters=[
                  {
                      'import_path': 'chatterbot.logic.BestMatch',
                      'default_response': 'Sorry, I am unable to process your request. Please try again.',
                      'maximum_similarity_threshold': 0.90
                  }
              ])

trainer = ListTrainer(bot)
data = load_data('/home/g6/Desktop/chatbot/Symptom2Disease (1).csv')
for text, label in data:
    trainer.train([text, label])

def ask_question(question):
    print(f'Clara: {question}')
    response = input().strip().lower()
    print(f'You: {response}')
    if response == 'bye':
        return "end", None
    return "continue", response

print("Hello, I am Clara, your chatbot assistant for DermaScreen! Please report any symptoms you might be experiencing or 'Bye' to end the chat.")

while True:
    print("Clara: What are your symptoms?")
    symptom = input().strip().lower()
    print(f'You: {symptom}')
    if symptom == 'bye':
        print('Clara: It was great talking to you! Bye!')
        break

    response = bot.get_response(symptom)
    print(f"Clara: Based on your symptoms, you might have {response.text}.")

    # Chronic illness query
    result, chronic_response = ask_question("Do you suffer from any chronic illnesses? Please indicate 'yes' or 'no'.")
    if result == "end":
        break
    if 'yes' in chronic_response:
        result, illness = ask_question("What illness(es) do you have?")
        if result == "end":
            break
        result, treatment_response = ask_question(f"Are you currently under treatment for {illness}? Please indicate 'yes' or 'no'.")
        if result == "end":
            break
        if 'yes' in treatment_response:
            result, medication = ask_question("What medications are you taking?")
            if result == "end":
                break
            result, duration = ask_question(f"How long have you been taking {medication}?")
            if result == "end":
                break
            result, dosage = ask_question(f"What is the dosage and frequency of administration for {medication}?")
            if result == "end":
                break

    # Allergy checks
    result, allergy_response = ask_question("Do you have any food or medication allergies? Please indicate 'yes' or 'no'.")
    if result == "end":
        break
    if 'yes' in allergy_response:
        result, allergies = ask_question("What are your allergies?")
        if result == "end":
            break
        result, reaction_response = ask_question("Have you had any allergic reactions recently?")
        if result == "end":
            break


    result, recent_med_response = ask_question("Have you been on any medication recently? Please indicate 'yes' or 'no'.")
    if result == "end":
        break
    if 'yes' in recent_med_response:
        result, recent_med = ask_question("What medication were you recently taking?")
        if result == "end":
            break
        result, recent_duration = ask_question(f"How long have you been taking {recent_med}?")
        if result == "end":
            break
        result, recent_dosage = ask_question(f"What is the dosage and frequency of administration for {recent_med}?")
        if result == "end":
            break

    additional_info = ask_question("Is there anything else you would like to add or discuss? (type 'Bye' to end): ")
    if additional_info[0] == 'end' or 'bye' in additional_info[1]:
        print('Clara: It was great talking to you! Bye!')
        break
