from dotenv import load_dotenv
import os
import torch
from xlm_roberta_base import XLMRobertaBase

model = XLMRobertaBase()

results = model.generate_question("Địa điểm du lịch Hạ Long Quảng Ninh là một trong những điểm đến hấp dẫn bậc nhất nước ta. Với diện tích lên đến 1.553km2 bao gồm 1.969 hòn đảo đá vôi mang nhiều hình thù đẹp mắt, sinh động. ")
print(results)