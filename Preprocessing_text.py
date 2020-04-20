import numpy as np
import pandas as pd 
import tensorflow as tf
import re
import time
import io
import os
path_to_zip = tf.keras.utils.get_file(
    'cornell movie-dialogs corpus.zip', origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)

path_to_lines = os.path.dirname(path_to_zip)+"/cornell movie-dialogs corpus/movie_lines.txt"
path_to_conv = os.path.dirname(path_to_zip)+"/cornell movie-dialogs corpus/movie_conversations.txt"

lines = io.open(path_to_lines,encoding = 'utf-8',errors='ignore').read().strip().split('\n')
conv_lines = io.open(path_to_conv,errors = 'ignore').read().strip().split('\n')

id2line ={}
for line in lines:
  _line = line.split(' +++$+++ ')
  if len(_line) ==5:
      id2line[_line[0]] = _line[4]

convs = []
for line in conv_lines:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace(r"'","").replace(" ","")
    convs.append(_line.split(','))

#question and answer
question = []
answer = []
for k in convs:
  for l in range(len(k)-1):
    question.append(id2line[k[l]])
    answer.append(id2line[k[l+1]])

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = " ".join(text.split())
    return text

clean_questions = []
for aquestion in question:
    clean_questions.append(clean_text(aquestion))
    
clean_answers = []    
for aanswer in answer:
    clean_answers.append(clean_text(aanswer))

min_length = 2
max_length =20
l = 0
short_question_temp = []
short_ans_temp = []
for aquestion in clean_questions:
  if len(aquestion)>=min_length and len(aquestion)<=max_length:
    short_question_temp.append(aquestion)
    short_ans_temp.append(clean_answers[l])
  l+=1

short_ans = []
short_question = []
m = 0
for aanswer in short_ans_temp:
  if len(aanswer)>=min_length and len(aanswer)>=max_length:
    short_ans.append(aanswer)
    short_question.append(short_question_temp[m])
  m+=1

for i in range(len(short_question)):
  short_question[i]='<Start>'+short_question[i]+'<End>'
for i in range(len(short_ans)):
  short_ans[i] = '<Start>'+short_ans[i]+'<End>'

#tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

tokenizer.fit_on_texts(short_question)
question_seq = tokenizer.texts_to_sequences(short_question)
question_seq = tf.keras.preprocessing.sequence.pad_sequences(question_seq,padding='post')
tokenizer.fit_on_texts(short_ans)
ans_seq = tokenizer.texts_to_sequences(short_ans)
ans_seq = tf.keras.preprocessing.sequence.pad_sequences(ans_seq, padding = 'post')