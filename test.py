#  统计token 长度
import json

L = []
# with open('data/mrc-ner.train', encoding='utf-8') as f:
#     res = json.load(f)
#     for r in res:
#         context_len = len(r['context'])
#         label_len = len(r['query'])
#         L.append(context_len + label_len)
#
#
# with open('len.json', 'w') as f:
#     json.dump(L, fp=f)

with open('len.json') as f:
    L = json.load(f)

L.sort()
print(L)
print(L[-1])
'''
满足90%长度的174
满足80%长度 142
'''

