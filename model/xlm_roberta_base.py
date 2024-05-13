import torch
import transformers
import pandas
import os
from dotenv import load_dotenv

class Custom_XLMRobertaBase(transformers.RobertaForMaskedLM):
  def __init__(self, config, state_dict):
    super(Custom_XLMRobertaBase, self).__init__(config)
    super().load_state_dict(state_dict)

  def forward(self, **kwargs):
    return super().forward(**kwargs)


class XLMRobertaBase:
    def __init__(self):
        # self._load_env()
        self.checkpoint = 'xlm-roberta-base'
        self.config = transformers.AutoConfig.from_pretrained(self.checkpoint)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.checkpoint)
        save = torch.load(r'D:\PC\University\NCKH\Code\model\xlm_roberta_base.pth', map_location=torch.device('cpu'))
        self.model = Custom_XLMRobertaBase(self.config, save['model'])

    def _load_env(self):
        load_dotenv(r'../.env')
        keys = os.environ.keys()
        env = os.environ['XLM_ROBERTA_BASE_PATH']
        print(1)

    def generate_question(self, paragragh):
        results = []
        for ct in paragragh.split('. '):
            if len(ct) < 10:
                continue
            qs = ' '
            while qs[-1] != '?' and len(qs) < 100:
                input = """
              Task: Generate question using a context and a answer below:
              Context: {ct}
              Answer: {ans}
              Question: {s}<mask>
                      """.format(ct=ct, ans=ct, s=qs)
                input = self.tokenizer(input, add_special_tokens=True, return_tensors='pt')
                o = self.model(**input)
                w = self.tokenizer.decode(o.logits[:, -2, :].argmax(dim=-1).tolist()[0])
                qs += ' ' + w
            results.append(qs)
        return results
