from config import START_TOKEN, END_TOKEN, IGNORE_LABEL

def gen_lines(filename):
    with open(filename) as f:
        for line in f:
            yield [START_TOKEN] + line.split() + [END_TOKEN]
