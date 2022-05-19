fr = open('./debug.train', 'r')
fw = open('./debug.dev', 'w')
for entry in fr.read().strip().split('\n\n'):
    text = [line.strip().split('\t')[0] for line in entry.split('\n')]
    for t in text:
        fw.write(t + '\t' + 'O' + '\t' + 'O' + '\n')
    fw.write('\n')