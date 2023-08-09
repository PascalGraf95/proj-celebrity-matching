import json

def save_dict_to_file(dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False)

def load_dict_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    my_dict = {
        "Abby Wambach": "Bekannte ehemalige US-amerikanische Fußballspielerin und Olympiasiegerin.",
        "Abigail Breslin": "Bekannte US-amerikanische Schauspielerin, bekannt aus Filmen wie 'Little Miss Sunshine'.",
        "Adam Peaty": "Britischer Schwimmer und Weltrekordhalter in der Brustschwimm-Distanz.",
        "Adel Tawil": "Deutscher Sänger und Mitglied der Band 'Ich + Ich'.",
        "Adele Neuhauser": "Österreichische Schauspielerin, bekannt für ihre Rolle in der Fernsehserie 'Tatort'.",
        "Aidan Alexander": "US-amerikanischer Schauspieler und Internetpersönlichkeit.",
        "Aidan Gallagher": "Bekannter US-amerikanischer Schauspieler aus der Serie 'The Umbrella Academy'.",
        "Albert Einstein": "Berühmter theoretischer Physiker, Entwickler der Relativitätstheorie.",
        "Alberto Contador": "Ehemaliger spanischer Radrennfahrer und Sieger mehrerer Grand Tours.",
        "Alejandro Valverde": "Spanischer Radrennfahrer, ehemaliger UCI Straßen-Weltmeister.",
        "Alex Morgan": "US-amerikanische Fußballspielerin und Weltmeisterin.",
        "Alexa Curtis": "Australische Sängerin und Gewinnerin von 'The Voice Kids Australia'.",
        "Alice Schwarzer": "Bekannte deutsche Feministin und Journalistin.",
        "Alicia von Rittberg": "Deutsche Schauspielerin, bekannt für ihre Filmrollen.",
        "Alligatoah": "Deutscher Rapper und Produzent."
}

    save_dict_to_file(my_dict, "celebrity_dictionary.txt")