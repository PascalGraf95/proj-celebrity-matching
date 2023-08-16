import os


if __name__ == '__main__':
    name_list = [f for f in os.listdir("Supportset_Celebrities")]
    # print(name_list)
    for i in range(0, len(name_list), 15):
        print("Gib mir eine Kurzbeschreibung folgender Personen in einem Satz  ohne den jeweiligen Namen zu nennen als Python Dictionary:")
        for i2 in range(i, i+15):
            print(f"{name_list[i2]}, ", end="")
        input(" ")
