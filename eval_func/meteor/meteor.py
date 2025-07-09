#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
import re
import os
import sys
import json
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


# print METEOR_JAR

class Meteor:

    def __init__(self):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=os.path.dirname(os.path.abspath(__file__)), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)
        
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'LEVIR_CC', 'vocab.json')
        with open(vocab_path, 'r') as f:
            word_vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in word_vocab.items()}


        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):

        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in range(len(res)):
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR

        self.meteor_p.stdin.write(eval_line + '\n')

        # Collect segment scores
        for i in range(len(res)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis, references):
        """ hypothesis and references are tokenIDs. """
        # Decode token IDs to strings if needed
        hypothesis_str = clean_string(self.decode_token_list(hypothesis))
        reference_list = [clean_string(self.decode_token_list(ref)) for ref in references]


        # print("Decoded hypothesis:", hypothesis_str)
        # print("Decoded refs:", reference_list)

        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        # print("PRİNTİNG score_line:")
        # print(score_line)
        # print("\n\n")
        self.meteor_p.stdin.write(score_line + '\n')
        self.meteor_p.stdin.flush()  # ensure it gets written
        # error_output = self.meteor_p.stderr.readline()
        # if error_output:
        #     print("JAVA STDERR:", error_output)

        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()

    def decode_token_list(self, token_list):
        # Handle case where token_list is a string of IDs like "452 211 23"
        if isinstance(token_list, str):
            try:
                token_ids = list(map(int, token_list.strip().split()))
                return ' '.join([self.inv_vocab.get(t, '<UNK>') for t in token_ids])
            except ValueError:
                # It's probably already a sentence
                return token_list
        elif isinstance(token_list, list):
            return ' '.join([self.inv_vocab.get(t, '<UNK>') for t in token_list])
        else:
            raise TypeError("Unsupported type for token_list: " + str(type(token_list)))


def clean_string(s):
    s = s.encode('ascii', 'ignore').decode('ascii')  # remove non-ascii
    s = re.sub(r'\s+', ' ', s)  # collapse multiple spaces
    s = s.strip()
    return s

