from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from mosestokenizer import *
from indicnlp.tokenize import sentence_tokenize

# this import might not work without restarting runtime (just restart (clear outputs) and then execute this cell only)
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils

from __future__ import unicode_literals, print_function
from spacy.lang.en import English
from transformers import AutoTokenizer
import os

#%cd ./indicTrans
# load the tranlation model from that directory
# from indicTrans.inference.engine import Model # because of this import, we have to do cd...
# en2indic_model = Model(expdir='/kaggle/working/en-indic')
# en2indic_model

root_path = '/content/drive/MyDrive/chaii/enhance_datasets/'

#data_to_load = '/content/drive/MyDrive/chaii/enhance_datasets/quoref.csv'
data_to_load = f'{root_path}newsqa.csv'
#file_name_to_save = 'quoref_tamil'
file_name_to_save = 'newsqa_tamil'

# let's load the sample squad dataset....
import pandas as pd

df = pd.read_csv(
    data_to_load,
    usecols=["context", "question", "id", "answer_text", "answer_start"],
    #nrows = 2**10, # sample
)
#df.drop_duplicates(subset=["context"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.shape)

# items to translate
en_context = list(df["context"].values)
en_question = list(df["question"].values)
en_answer_text = list(df["answer_text"].values)


# writing the paragraphs line-by-line

import io
from tqdm.auto import tqdm



def get_file_lines(file):
  with open(file,'r') as temp_open:
    os.fsync(temp_open)
    return len([line for line in temp_open.readlines() if len(line.strip()) > 0])


def batched_translate(input_file, output_file):
    import tqdm
    import os

    batch_count = 500
    translation_result = []
    total_processed = 0
    tmp_file_no = 0

    with open(input_file, 'r') as input:
        lines = input.readlines()
        batch = []
        for iline, line in tqdm.tqdm(enumerate(lines), total=len(lines)):
            line = line.strip()
            batch.append(line)
            if len(batch) >= batch_count or iline == len(lines) - 1:
                # write batch
                with open(f'{input_file}.{tmp_file_no}.tmp.txt', 'w') as output:
                    for bat in batch:
                        output.write(f'{bat}\n')
                    os.fsync(output)
                total_processed += len(batch)
                batch_lines = len(batch)
                batch = []
                # translate batch

                for iretry in range(5):
                    !./ joint_translate_updated.sh
                    {input_file}.
                    {tmp_file_no}.tmp.txt
                    {output_file}.
                    {tmp_file_no}.tmp.txt
                    "en" "ta" '../en-indic'
                    if get_file_lines(f'{output_file}.{tmp_file_no}.tmp.txt') > 0:
                        break
                    else:
                        print(f'{output_file}.{tmp_file_no}.tmp.txt error, retry {iretry} times...')

                print("batch line counts:")
                assert get_file_lines(f'{input_file}.{tmp_file_no}.tmp.txt') == \
                       get_file_lines(f'{output_file}.{tmp_file_no}.tmp.txt'), \
                    f"generated lines{len(temp_open.readlines())} should match batch:{batch_lines}"

                tmp_file_no += 1

                '''
                # read batch
                with open(f'ta_paragraphs_splitted.tmp.{iline}.txt', 'r') as trans:
                  lines = trans.readlines()
                  #translation_result.extend(lines)
        
                  for line in lines:
                    line=line.strip()
                    final_output.write(f'{line}\n')
                  os.fsync(final_output)
                  print(f"accumulate input count: {total_processed} and accumulate output line count:")
                  !wc -l ta_paragraphs_splitted.txt
                '''

    # for line in translation_result:
    print(f"tmp files gen'ed: {tmp_file_no}")

    return tmp_file_no

def split_tans_then_combine(to_split_file, output_file):
  # split...


  tokenizer = AutoTokenizer.from_pretrained('deepset/xlm-roberta-large-squad2')

  original_ids = []
  sents = []
  nlp = English()
  nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated

  with io.open(to_split_file, "r") as f_ptr:
      #en_context = [line.strip() for line in f_ptr]

      for i, line in enumerate(f_ptr):
          doc = nlp(line.strip())
          to_add = ''
          to_add_token_count = 0
          for sent in doc.sents:
              current = sent.string.strip()
              current_count = len(tokenizer(current)['input_ids'])

              if to_add_token_count + current_count > 200 and to_add_token_count > 0:
                  sents.append(to_add)
                  original_ids.append(i)
                  to_add = current
                  to_add_token_count = current_count
              else:
                  to_add = f"{to_add} {current}"
                  to_add_token_count += current_count
          sents.append(to_add)
          original_ids.append(i)

  with io.open(f"splitted_{to_split_file}", "w") as f_ptr:
    for sent in sents:
      f_ptr.write(f'{sent}\n')
    os.fsync(f_ptr)

  !wc -l splitted_{to_split_file}

  tmp_file_no = batched_translate(f"splitted_{to_split_file}", f"splitted_{output_file}")

  # combine batched tmps
  import tqdm
  with io.open(f"splitted_{output_file}", "w") as splitted_combine:
    for i in tqdm.tqdm(range(tmp_file_no)):
      with open(f'splitted_{output_file}.{i}.tmp.txt','r') as temp_open:
        lines = temp_open.readlines()
        for line in lines:
          splitted_combine.write(f'{line.strip()}\n')
    os.fsync(splitted_combine)

  # combine splitted
  # recover non-split version...
  original_lines = []
  previous_id = -1
  current_line_list = []
  with io.open(f"splitted_{output_file}", "r") as f_ptr:
      for i, line in enumerate(f_ptr):
        original_id = original_ids[i]
        #en_context = [line.strip() for line in f_ptr]
        if previous_id == -1 or original_id == previous_id:
          current_line_list.append(line.strip())
        else:
          original_lines.append(' '.join(current_line_list))
          current_line_list = []
          current_line_list.append(line.strip())
          assert original_id == len(original_lines)

        previous_id = original_id

  original_lines.append(' '.join(current_line_list))


  with io.open(output_file, "w") as f_ptr:
    for line in original_lines:
      f_ptr.write(f'{line}\n')
    os.fsync(f_ptr)


with io.open(f"{file_name_to_save}_en_paragraphs.txt", "w") as f_ptr:
    for line in en_context:
        f_ptr.write(line.replace("\n", "").strip())
        f_ptr.write("\n")

with io.open(f"{file_name_to_save}_en_question.txt", "w") as f_ptr:
    for line in en_question:
        f_ptr.write(line.replace("\n", "").strip())
        f_ptr.write("\n")

with io.open(f"{file_name_to_save}_en_answer_text.txt", "w") as f_ptr:
    for line in en_answer_text:
        f_ptr.write(line.strip())
        f_ptr.write("\n")


split_tans_then_combine(f"{file_name_to_save}_en_paragraphs.txt", f"{file_name_to_save}_ta_paragraphs.txt")

#split_tans_then_combine(f"{file_name_to_save}_en_question.txt", f"{file_name_to_save}_ta_question.txt")

split_tans_then_combine(f"{file_name_to_save}_en_answer_text.txt", f"{file_name_to_save}_ta_answer_text.txt")

assert get_file_lines(f"{file_name_to_save}_en_paragraphs.txt") == get_file_lines(f"{file_name_to_save}_ta_paragraphs.txt")
assert get_file_lines(f"{file_name_to_save}_en_question.txt") == get_file_lines(f"{file_name_to_save}_ta_question.txt")
assert get_file_lines(f"{file_name_to_save}_en_answer_text.txt") == get_file_lines(f"{file_name_to_save}_ta_answer_text.txt")

# en_context, en_question, en_answer_text -- for english stuff

# read the context from the file "ta_paragraphs.txt"
with io.open(f"{file_name_to_save}_ta_paragraphs.txt", "r") as f_ptr:
    ta_context = [line.strip() for line in f_ptr]

# read the context from the file "ta_question.txt"
with io.open(f"{file_name_to_save}_ta_question.txt", "r") as f_ptr:
    ta_question = [line.strip() for line in f_ptr]

# read the context from the file "ta_answer_text.txt"
with io.open(f"{file_name_to_save}_ta_answer_text.txt", "r") as f_ptr:
    ta_en_answer_text = [line.strip() for line in f_ptr]

cnt = 0
good_ids = set()

USE_ALL_IDS = False
for idx in range(len(ta_en_answer_text)):
    if USE_ALL_IDS or ta_en_answer_text[idx] in ta_context[idx]:
        # basically naswer is present as-is in the translated text
        cnt += 1
        good_ids.add(idx)
cnt / len(ta_en_answer_text), len(good_ids) # 22% is not bad!

from dataclasses import dataclass

@dataclass
class TamilExtraData:
    ta_context :str
    ta_question :str
    ta_answer_text :str
    en_context :str
    en_question :str
    en_answer_text :str

final_data = []
for idx in good_ids:
    final_data.append(
        TamilExtraData(
            ta_context=ta_context[idx], ta_answer_text=ta_en_answer_text[idx], ta_question=ta_question[idx],
            en_context=en_context[idx], en_answer_text=en_answer_text[idx], en_question=en_question[idx],
        ).__dict__
    )

df = pd.DataFrame.from_records(final_data)
df["answer_start"] = df[["ta_context", "ta_answer_text"]].apply(lambda row: row[0].find(row[1]), axis=1)

df = df.rename(columns={'ta_context':'context',
                   'ta_question':'question',
                   'ta_answer_text':'answer_text',
                   })
df = df[['context', 'question','answer_text','answer_start']]
df['id'] = df.index.values
df['id'] = df['id'].apply(lambda x: f"quoref_{x}")
df['language']='tamil'


df.to_csv(f"{root_path}{file_name_to_save}.csv", index=None) # almost extra 1000 tamil dataset rows... Enjoy People...


print(f"{root_path}{file_name_to_save}.csv gen'd.")