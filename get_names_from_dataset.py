import os
from load_write_dict import *


if __name__ == '__main__':
    cel_dict = load_dict_from_file("celebrity_dictionary.txt")
    name_list = [f for f in os.listdir("Supportset_Celebrities")]
    # print(name_list)
    for i in range(0, len(name_list), 100):
        print("Gib mir eine Kurzbeschreibung folgender Personen in einem Satz  ohne den jeweiligen Namen zu nennen als Python Dictionary:")
        for i2 in range(i, i+100):
            if name_list[i2] not in cel_dict.keys():
                print(f"{name_list[i2]}, ", end="")
        input(" ")
