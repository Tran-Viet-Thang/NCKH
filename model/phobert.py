import transformers
import torch
import pandas as pd
import os
from dotenv import load_dotenv
import py_vncorenlp


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, vncorenlp):
        self.contexts = df['ct']
        self.answers = df['ans']
        self.questions = df['qs']

        self.vncorenlp = vncorenlp
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.contexts)

    def annotate(self, sentence):
        sentence_dict = self.vncorenlp.annotate_text(sentence)[0]

        store_idx = []
        store_insert = []

        for idx, w in enumerate(sentence_dict):
            token = self.tokenizer.encode(w['wordForm'], add_special_tokens=False)
            if len(token) != 1:
                store_idx.append(idx)
                s = w['wordForm'].replace('_', ' ')
                insert = []
                for af_w in s.split(' '):
                    insert.append({'index': idx,
                                   'wordForm': af_w,
                                   'posTag': w['posTag'],
                                   'nerLabel': w['nerLabel'],
                                   'head': w['head'],
                                   'depLabel': w['depLabel']})
                store_insert.append(insert)

        for idx, insert in zip(store_idx[::-1], store_insert[::-1]):
            sentence_dict = sentence_dict[:idx] + insert + sentence_dict[idx + 1:]

        cat = lambda key: ' '.join([s[key] for s in sentence_dict])
        sentence = cat('wordForm')
        pos = cat('posTag')
        ner = cat('nerLabel')
        dep = cat('depLabel')
        return sentence, pos, ner, dep

    def __getitem__(self, idx):
        context = self.contexts[idx]
        answer = self.answers[idx]
        question = self.questions[idx]

        context = context.replace('  ', ' .')
        answer = answer.replace('  ', ' .')

        rm_space = lambda s: ' '.join([w for w in s.split(' ') if w != ''])
        rm = lambda s: rm_space(s).replace('_', ' ')

        context = rm(context)
        answer = rm(answer)
        question = rm(question)

        count = lambda c, a: [1 if w in c else 0 for w in a.split(' ')]
        similar = lambda c, a: sum(count(c, a)) / len(count(c, a))

        answer = self.vncorenlp.annotate_text(answer)
        cat = lambda at: ' '.join([x['wordForm'] for x in at])

        sentences = [cat(answer[idx]) for idx in answer]
        similar_v = [similar(s, question) for s in sentences]
        max_i = similar_v.index(max(similar_v))

        answer = sentences[max_i].replace('_', ' ')

        context = self.vncorenlp.annotate_text(context)
        cat = lambda at: ' '.join([x['wordForm'] for x in at])

        sentences = [cat(context[idx]) for idx in context]
        similar_v = [similar(s, question) for s in sentences]
        max_i = similar_v.index(max(similar_v))

        context = sentences[max_i].replace('_', ' ')

        extract = lambda sentence: {f'{key}': value for key, value in
                                    zip(['s', 'p', 'n', 'd'], list(self.annotate(sentence)))}

        context_dict = extract(context)
        answer_dict = extract(answer)
        question_dict = extract(question)

        return context_dict, answer_dict, question_dict

class Custom_PhoBert(transformers.RobertaForMaskedLM):
  def __init__(self, config, state_dict):
    super(Custom_PhoBert, self).__init__(config)
    super().load_state_dict(state_dict)

  def forward(self, input):
    o = super().forward(**input)
    return o

class Phobert:
    def __init__(self):
        self._load_env()
        self.vncorenlp = py_vncorenlp.VnCoreNLP(
            save_dir=r'D:\PC\University\NCKH\Code\model\vncorenlp'
        )
        self.checkpoint = 'vinai/phobert-base-v2'
        self.config = transformers.AutoConfig.from_pretrained(self.checkpoint)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = Custom_PhoBert(self.config, torch.load(r'D:\PC\University\NCKH\Code\model\phobert.pth')['model'])

    def _load_env(self):
        load_dotenv(r'../.env')

    def generate_question(self, paragragh):
        results = []
        a_t = self.vncorenlp.annotate_text(paragragh)
        p_loc = []
        for key in a_t.keys():
            for w in a_t[key]:
                if 'LOC' in w['nerLabel'] and '_' in w['wordForm']:
                    p_loc.append(w['wordForm'])

        sens = paragragh.split('. ')

        for idx, sen in enumerate(sens):
            if sen == '':
                continue
            context = sen
            answer = sen
            question = '.'
            df_t = pd.DataFrame({'ct': [context], 'ans': [answer], 'qs': [question]})
            dataset_t = Dataset(df_t, self.tokenizer, self.vncorenlp)

            lst_question = set()
            r = 0
            while r < 20 and len(lst_question) < 1:

                context, answer, question = dataset_t[0]
                context = context['s']
                answer = answer['s']

                input = f'<s> {context} </s> {answer} </s> <mask>'
                pred_q = ''
                with torch.no_grad():
                    w = ''
                    c = 0
                    while w != '?' and c < 20:
                        tokens = self.tokenizer(input, add_special_tokens=False, return_tensors='pt')
                        o = self.model(tokens)
                        w = self.tokenizer.decode(o.logits[:, -1, :].argmax(dim=1).tolist())
                        input = input.replace('<mask>', w)
                        pred_q += w + ' '
                        input += ' <mask>'
                        c += 1

                r += 1

                if '<mask>' in pred_q:
                    continue

                a_t = self.vncorenlp.annotate_text(pred_q.replace('_', ' '))
                q_loc = []
                for key in a_t.keys():
                    for w in a_t[key]:
                        if 'LOC' in w['nerLabel'] and '_' in w['wordForm']:
                            q_loc.append(w['wordForm'])
                for loc in q_loc:
                    if loc not in p_loc and len(p_loc) > 0:
                        pred_q = pred_q.replace(loc, p_loc[0])

                if pred_q[-2] == '?':
                    lst_question.add(pred_q)

            results += list(lst_question)

        return results
