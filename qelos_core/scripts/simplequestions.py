import torch
import qelos_core as q
import os
import pickle
import tqdm


def load_data(p="../../datasets/simplequestions/"):
    tt = q.ticktock("dataloader")
    tt.tick("loading")
    questions, subjects, subject_names, relations, spans, (start_valid, start_test) \
        = load_questions(p)
    generate_candidates(p)
    tt.tock("{} questions loaded".format(len(questions)))

    tt.tick("generating matrices")
    qsm = q.StringMatrix(freqcutoff=2)
    qsm.tokenize = lambda x: x.split()
    for question in tqdm.tqdm(questions[:start_valid]):
        qsm.add(question)
    qsm.unseen_mode = True
    for question in tqdm.tqdm(questions[start_valid:]):
        qsm.add(question)
    tt.msg("finalizing")
    qsm.finalize()
    print(qsm[0])
    q.embed()
    tt.tock("matrices generated")


def load_questions(p):
    questions = []
    subjects = []
    subject_names = []
    relations = []
    spans = []
    q_ids = []
    i = 0
    start_valid = None
    start_test = None
    for line in tqdm.tqdm(open(os.path.join(p, "processed", "all.txt"))):
        splits = line.split("\t")
        q_ids.append(splits[0])
        subjects.append(splits[1])
        subject_names.append(splits[2])
        relations.append(splits[3])
        questions.append(splits[5])
        spans.append(splits[6])
        if "valid" in splits[0]:
            start_valid = i
        elif "test" in splits[0]:
            start_test = i
        i += 1
    return questions, subjects, subject_names, relations, spans, (start_valid, start_test)


def generate_candidates(p):
    pass    # TODO



def run(lr=0.001):
    load_data()


if __name__ == '__main__':
    q.argprun(run)