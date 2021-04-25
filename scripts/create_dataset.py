import os, sys
import pickle
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english'))
stop_words |= {"'", '"', ':', ';', '.', ',', '-', '--', '...', '//', '/', '!', '?', "'s", "@", "<p>", "(", ")"}
print(stop_words)

def unzip():
    l = os.listdir("./text")
    for tar in l:
        os.system("unzip {} -d {}".format("../COHA/text/"+tar, "../COHA/unzipped_text/"))

def create_dataset(arch=False, year_start=1810, year_end=2009, time_interval=20, validation=False):
    validation_id = (year_end-year_start)//time_interval
    if not arch:
        target_words = ["band", "power", "transport", "bank"]
        snippets = {i: {target_words[j]: [] for j in range(len(target_words))} for i in range(year_start, year_end+1)}
        count = {target_words[i]: 0 for i in range(len(target_words))}
        files = os.listdir("../COHA/unzipped_text/")
        for fn in files:
            with open("../COHA/unzipped_text/"+fn) as f:
                f.readline()
                f.readline()
                text_raw = f.readline().split()
                text = [w.lower() for w in text_raw if not w.lower() in stop_words]
                year = int(fn.split("_")[1])
                for i, word in enumerate(text):
                    if word in target_words:
                        if i < 5 or i > len(text) - 6: continue
                        snippets[year][word].append(text[i-5:i]+text[i+1:i+6])
                        count[word] += 1
        with open("./snippets.pickle", "wb") as f:
            pickle.dump(snippets, f)
    else:
        with open("./snippets.pickle", "rb") as f:
            snippets = pickle.load(f)

    if validation:
        for year, corpus in snippets.items():
            for tar_word, documents in corpus.items():
                for snippet in documents:
                    if not os.path.exists("../data/{}".format(tar_word)):
                        os.mkdir("../data/{}".format(tar_word))
                    if (year-year_start)//time_interval == validation_id:
                        with open("../data/{}/valid.txt".format(tar_word), "a") as f:
                            f.write(" ".join(snippet)+"\n")
                        with open("../data/{}/valid.label".format(tar_word), "a") as f:
                            f.write(str((year-year_start)//time_interval)+"\n")
                    else:
                        with open("../data/{}/train.txt".format(tar_word), "a") as f:
                            f.write(" ".join(snippet)+"\n")
                        with open("../data/{}/train.label".format(tar_word), "a") as f:
                            f.write(str((year-year_start)//time_interval)+"\n")
    else:
        for year, corpus in snippets.items():
            for tar_word, documents in corpus.items():
                for snippet in documents:
                    if not os.path.exists("../data/{}".format(tar_word)):
                        os.mkdir("../data/{}".format(tar_word))
                    with open("../data/{}/documents.txt".format(tar_word), "a") as f:
                        f.write(" ".join(snippet)+"\n")
                    with open("../data/{}/time_labels.txt".format(tar_word), "a") as f:
                        f.write(str((year-year_start)//time_interval)+"\n")

create_dataset()