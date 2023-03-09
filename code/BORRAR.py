import json
import pickle
with open("extra/true_fact.txt", 'rb') as fp:
    person = pickle.load(fp)


print(person)