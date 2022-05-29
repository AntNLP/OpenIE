from faulthandler import disable
import pandas as pd
import argparse
dic = {}
mx_len = 0
tot = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usage for OPENIE.")
    parser.add_argument('--input' , type=str, help="Path to source file.")
    parser.add_argument('--output', type=str, help="Path to target file.")
    args = parser.parse_args()
    fr = open(args.input, 'r')
    fw = open(args.output, 'w')
    entries = fr.read().strip().split('\n\n')
    for entry in entries:
        text = " ".join([line.strip().split('\t')[1] for line in entry.split('\n')])
        if not (text in dic)  :
            dic[text] = {
                'text': [line.strip().split('\t')[1] for line in entry.split('\n')],
                'pre_tag': ['O' for line in entry.split('\n')],
                'extractions': []
            }
        arg_tag = []
        pre_tag = []
        cnt = 0
        for line in entry.split('\n'):
            tag = line.strip().split('\t')[-1]
            if(tag[0] == 'A'):
                if(tag[0] == 'A' ):
                    if(tag[1] == '0' or tag[1] == '1' or tag[1] == '2' or tag[1] == '3'):
                        arg_tag.append(tag)
                    else:
                        arg_tag.append('O')
                else:
                    arg_tag.append('O')
                pre_tag.append('O')
            elif(tag[0] == 'P'):
                if(tag[-1] == 'I'):
                    pre_tag.append('P-I')
                    dic[text]['pre_tag'][cnt] = 'P-I'
                else:
                    pre_tag.append('P-B')
                    dic[text]['pre_tag'][cnt] = 'P-B'
                arg_tag.append('O')
            else:
                arg_tag.append('O')
                pre_tag.append('O')
            cnt+=1
        dic[text]['extractions'].append(
            {
                'arg_tag': arg_tag, 
                'pre_tag': pre_tag
            }
        ) 
    for k, v in dic.items():
        # train/dev/test data
        text = v['text']
        tag = v['pre_tag']
        ext = []
        for extraction in v['extractions']:
            spans = []
            tags = extraction['arg_tag']
            p_tags = extraction['pre_tag']
            for i in range(len(tags)):
                if(p_tags[i] != 'O'):
                    tags[i] = p_tags[i]
            ext.append(tags)
        # spacy or sth
        mx_len = max(mx_len, len(text))
        for i in range(len(tag)):
            fw.write(text[i] + '\t' + tag[i])
            for j in range(len(ext)):
                fw.write('\t' + ext[j][i])
            fw.write('\n')
        fw.write('\n')
print(mx_len)




