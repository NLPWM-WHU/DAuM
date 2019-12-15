# -*- coding: utf-8 -*-
import cPickle as pkl
import numpy as np
import random
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def word_fre(pathlist):
    word_fre_dict = {}
    n = 0
    for path in pathlist:
        for line in open(path, 'r'):
            strs = line.strip().split('\t\t', 3)
            doc = strs[3]
            words = doc.split(' ')
            # for i in range(len(words)):
            #     if words[i] in word_fre_dict.keys():
            #         count = word_fre_dict[words[i]]
            #         word_fre_dict[words[i]] = count + 1
            #     else:
            #         word_fre_dict[words[i]] = 1
            for word in words:
                try:
                    count = word_fre_dict[word]
                    if count <= 5:
                        word_fre_dict[word] = count + 1
                except KeyError:
                    word_fre_dict[word] = 1
            n += 1
            print n
    return word_fre_dict

def filter_low_fre(pathlist, filterlist, wfd):
    for i, path in enumerate(pathlist):
        f = open(filterlist[i], 'w')
        for line in open(path, 'r'):
            strs = line.strip().split('\t\t', 3)
            f.write(strs[2] + '\t\t')
            doc = strs[3]
            words = doc.split(' ')
            boo = False
            for word in words:
                if wfd[word] > 5:
                    f.write(word + ' ')
                    boo = True
                elif wfd[word] == 5:
                    f.write('unk ')
                    boo = True
            if not boo:
                f.write('unk ')
            f.write('\n')
        f.close()

def filter_dot(pathlist, filterlist):
    for i, path in enumerate(pathlist):
        f = open(filterlist[i], 'w')
        for line in open(path, 'r'):
            strs = line.strip().split('\t\t', 1)
            f.write(strs[0] + '\t\t')
            doc = strs[1]
            words = doc.split(' ')
            boo = False
            for word in words:
                if word != '.':
                    f.write(word + ' ')
                    if word != '<sssss>':
                        boo = True
            if not boo:
                f.write('unk ')
            f.write('\n')
        f.close()

def filter_emptysen(pathlist, filterlist):
    for i, path in enumerate(pathlist):
        f = open(filterlist[i], 'w')
        for line in open(path, 'r'):
            strs = line.strip().split('\t\t', 1)
            f.write(strs[0] + '\t\t')
            doc = strs[1]
            sens = doc.split('<sssss>')
            for sen in sens:
                if sen.strip() != '':
                    f.write(sen + ' ')
                    boo = True
            if not boo:
                f.write('unk ')
            f.write('\n')
        f.close()

# def sort_by_sen_length(trainset, sortedtrain):
#     index_len = {}
#     index_doc = {}
#     n = 0
#     for line in open(trainset, 'r'):
#         n += 1
#         index_doc[n] = line
#         strs = line.strip().split('\t\t',1)
#         if len(strs) < 2:
#             print n
#         doc = strs[1]
#         sens = doc.split('<sssss>')
#         index_len[n] = len(sens)
#     list = sorted(index_len.items(), key=lambda item: item[1])
#     f = open(sortedtrain, 'w')
#     for i in range(len(list)):
#         f.write(index_doc[list[i][0]])
#     f.close()

def sort_by_sen_length(trainset, sortedtrain):
    linelist = []
    for line in open(trainset, 'r'):
        linelist.append(line)
    lens = map(lambda x: len(x.split('<sss>')[0].split(' ')), linelist)
    index_len = zip(range(len(linelist)), lens)
    sortedlist = sorted(index_len, key=lambda item: item[1])
    f = open(sortedtrain, 'w')
    for i in range(len(sortedlist)):
        f.write(linelist[sortedlist[i][0]])
    f.close()


def gene_corpus(setlist, corpus):
    f = open(corpus, 'w')
    for set in setlist:
        for line in open(set, 'r'):
            strs = line.strip().split('\t\t')
            doc = strs[3]
            sens = doc.split('<sssss>')
            for word in sens:
                f.write(word)
            f.write('\n')
    f.close()

def build_dict(w2vfile, embeddingmatrix):
    n = 0
    f = open(embeddingmatrix, 'w')
    dict = {}
    for line in open(w2vfile, 'r'):
        n += 1
        if n > 2:
            strs = line.strip().split(' ', 1)
            dict[strs[0]] = n - 3
            f.write(strs[1] + '\n')
    f.close()
    return dict

def word2id(sourceset, idset, labelset, dict):
    for i, path in enumerate(sourceset):
        fid = open(idset[i], 'w')
        flabel = open(labelset[i], 'w')
        for line in open(path, 'r'):
            strs = line.strip().split('\t\t', 1)
            flabel.write(strs[0] + '\n')
            doc = strs[1]
            words = doc.split(' ')
            for word in words:
                if word == '<sssss>':
                    fid.write(word + ' ')
                else:
                    fid.write(str(dict[word]) + ' ')
                    # try:
                    #     fid.write(str(dict[word]) + ' ')
                    # except KeyError:
                    #     print word
            fid.write('\n')
        fid.close()
        flabel.close()


def preRes(prefile, truefile, resfile):
    all = 0
    same = 0
    with open(prefile, 'r') as f1, open(truefile, 'r') as f2:
        for line1 in f1:
            line2 = f2.readline()
            all += 1
            strsp = line1.strip().split('\t\t')
            strst = line2.strip().split('\t\t')
            pla = int(strsp[2]) - 1
            tla = int(strst[2])
            if pla == tla:
                same += 1
    acc = float(same)/all
    f = open(resfile, 'w')
    f.write('Num of test samples:' + str(all) + '\t' + 'correct prediction:' + str(same) + '\t' + 'accuracy:' + str(acc))

def get_idsen_dict(idfile, senfile):
    idsen = {}
    with open(idfile, 'r') as f1, open(senfile, 'r') as f2:
        for line1 in f1:
            line2 = f2.readline()
            idsen[line1.strip()] = line2.strip()
    return idsen


def get_sorted_xy(path, classnum, idsen, xfile, yfile):
    idlabel = {}
    idlen = {}
    for i in range(classnum):
        for line in open(path + str(i) + '.txt'):
            id = line.strip()
            idlabel[id] = str(i)
            sen = idsen[id]
            strs = sen.split(' ')
            idlen[id] = len(strs)
    list = sorted(idlen.items(), key=lambda item: item[1])
    fx = open(xfile, 'w')
    fy = open(yfile, 'w')
    for i in range(len(list)):
        fx.write(idsen[list[i][0]] + '\n')
        fy.write(idlabel[list[i][0]] + '\n')
    fx.close()
    fy.close()

def embedding_pre(glovefile, trainfile, devfile, testfile, embeddingfile, wdict):
    word = []
    for data in [trainfile, devfile, testfile]:
        for line in open(data, 'r'):
            review = line.strip().split('<sss>')[0]
            words = review.strip().split(' ')
            for w in words:
                # if w != '':
                #     # print review
                word.append(w)
    word = set(word)
    wordlen = len(word)
    print str(wordlen)
    arr = np.random.uniform(low=-0.01, high=0.01, size=(wordlen + 1, 300))
    count = 0
    f = open(wdict, 'w')
    for line in open(glovefile, 'r'):
        strs = line.strip().split(' ', 1)
        w = strs[0]
        if w in word:
            count += 1
            f.write(w + '\t' + str(count) + '\n')
            values = strs[1].split(' ')
            for i in range(len(values)):
                arr[count][i] = float(values[i])
            word.remove(w)
    for w in word:
        count += 1
        f.write(w + '\t' + str(count) + '\n')
    f.close()
    np.save(embeddingfile, arr)

def getDict(listfile):
    lines = map(lambda x : x.strip(), open(listfile).readlines())
    size = len(lines)
    d = dict([(item[0], item[1]) for item in zip(lines, xrange(size))])
    return d

def w2id(w, wdict):
    if w == 'null':
        return '0'
    else:
        try:
            return wdict[w]
        except:
            return -1

def pre(sourcedata, userDict, proDict, wordDict, x, y, u, p):
    lines = map(lambda x : x.split('\t\t'), open(sourcedata).readlines())
    label = map(lambda x: int(x[2]) - 1, lines)
    usr = map(lambda line: userDict[line[0]], lines)
    prd = map(lambda line: proDict[line[1]], lines)
    docs = map(lambda line: line[3], lines)
    docs = map(lambda sentence: sentence.strip().split(' '), docs)
    docs = map(lambda sentence: filter(lambda wordid: wordid != -1, map(lambda word: w2id(word, wordDict), sentence)), docs)
    for (d, f) in zip([label, usr, prd], [y, u, p]):
        fi = open(f, 'w')
        for i in range(len(d)):
            fi.write(str(d[i]) + '\n')
        fi.close()
    fi = open(x, 'w')
    for i in range(len(docs)):
        for j in range(len(docs[i])):
            fi.write(str(docs[i][j]) + ' ')
        fi.write('\n')
    fi.close()

def countOOV(testcor, wordlist):
    wl = []
    testwl = []
    for line in open(wordlist, 'r'):
        wl.append(line.strip())
    wl = set(wl)
    print len(wl)
    for line in open(testcor, 'r'):
        sentences = line.strip().split('\t\t')[1].split('<ssssss>')
        for sen in sentences:
            print sen
            words = sen.split(' ')
            for w in words:
                testwl.append(w)
    testwl = set(testwl)
    print len(testwl)
    dif = testwl - wl
    print len(dif)
    for w in testwl:
        if w not in wl:
            print w

def parseXML(XMLfile, output):
    tree = ET.parse(XMLfile)
    root = tree.getroot()
    # f = open(output, 'w')
    posi = 0
    nega = 0
    neu = 0
    con = 0
    flag = 0
    for child in root:
        review = child[0].text.lower()
        # f.write(review + '\t' + '<sss>' + '\t')
        if child[1].tag == 'aspectTerms':
            flag += 1
            if flag % 6 == 4:
                for subsub in child[1]:
                    po = subsub.attrib['polarity']

                    if po == 'conflict':
                        con += 1
                    if po == 'positive':
                        posi += 1
                    if po == 'negative':
                        nega += 1
                    if po == 'neutral':
                        neu += 1
                    # f.write(subsub.attrib['term'].lower() + ':' + po + '\t' + '<ss>' + '\t')
            # f.write('<sss>')
            # for subsub in child[2]:
            #     po = subsub.attrib['polarity']
            #
            #     # f.write(subsub.attrib['category'].lower() + ':' + po + '\t' + '<ss>' + '\t')
            # # f.write('\n')
        # else:
        #     # f.write('NULL' + '\t' + '<sss>')
        #     for subsub in child[1]:
        #         po = subsub.attrib['polarity']
        #         if po == 'conflict':
        #             con += 1
        #         if po == 'positive':
        #             posi += 1
        #         if po == 'negative':
        #             nega += 1
        #         if po == 'neutral':
        #             neu += 1
        #         # f.write(subsub.attrib['category'].lower() + ':' + po + '\t' + '<ss>' + '\t')
        #     # f.write('\n')
    # f.close()
    count = posi + nega + neu + con
    print str(posi) + ',' + str(nega) + ',' + str(neu) + ',' + str(con) + ',' + str(count)


def semEvalPre(tokenfile, termtrain, termval, asptrain, aspval):
    lines = map(lambda x: x.strip().split('<sss>'), open(tokenfile).readlines())
    reviews = map(lambda line: line[0].strip(), lines)
    terms = map(lambda line: line[1].strip().split('<ss>'), lines)
    aspects = map(lambda line: line[2].strip().split('<ss>'), lines)
    # tokenreviews = map(lambda x: x.strip(), open(tokenfile).readlines())
    # temptraindata = map(lambda x: x.strip().split('\t\t')[1], open(temptrain).readlines())
    traintuple = zip(reviews, terms, aspects)
    ftt = open(termtrain, 'w')
    ftv = open(termval, 'w')
    fat = open(asptrain, 'w')
    fav = open(aspval, 'w')
    posi = 0
    nega = 0
    neu = 0
    flag = 1
    fla = 1
    for i in range(len(traintuple)):
        review = traintuple[i][0]
        term = traintuple[i][1]
        aspect = traintuple[i][2]
        # print str(len(aspect))
        if term[0] != 'NULL':
            for te in term:
                if te != '':
                    strs = te.split(':')
                    text = strs[0].strip()
                    po = strs[1].strip()
                    if po != 'conflict':
                        if flag % 6 == 4:
                            ftv.write(review + '\t' + text + '\t' + po + '\n')
                        else:
                            ftt.write(review + '\t' + text + '\t' + po + '\n')
                        # ftt.write(review + '\t' + text + '\t' + po + '\n')
            for asp in aspect:
                if asp != '':
                    strs = asp.split(':')
                    text = strs[0].strip()
                    po = strs[1].strip()
                    if po != 'conflict':
                        if flag % 6 == 4:
                            fav.write(review + '\t' + text + '\t' + po + '\n')
                        else:
                            fat.write(review + '\t' + text + '\t' + po + '\n')
                        # fat.write(review + '\t' + text + '\t' + po + '\n')
            flag += 1
        else:
            for asp in aspect:
                if asp != '':
                    strs = asp.split(':')
                    text = strs[0].strip()
                    po = strs[1].strip()
                    if po != 'conflict':
                        if fla % 6 == 3:
                            fav.write(review + '\t' + text + '\t' + po + '\n')
                        else:
                            fat.write(review + '\t' + text + '\t' + po + '\n')
                        # fat.write(review + '\t' + text + '\t' + po + '\n')
            fla += 1
    ftt.close()
    ftv.close()
    fat.close()
    fav.close()
    all = posi + nega + neu
    print str(posi) + ',' + str(nega) + ',' + str(neu) + ',' + str(all)

def sentiment2id(senti):
    if senti == 'positive':
        return 0
    if senti == 'neutral':
        return 1
    if senti == 'negative':
        return 2

def cate2id(cate):
    # d = {"RESTAURANT #GENERAL":0, "RESTAURANT #PRICES":1, "RESTAURANT #MISCELLANEOUS":2,
    #      "FOOD #PRICES":3, "FOOD #QUALITY":4, "FOOD #STYLE_OPTIONS":5,
    #      "DRINKS #PRICES":6, "DRINKS #QUALITY":7, "DRINKS #STYLE_OPTIONS":8,
    #      "AMBIENCE #GENERAL":9, "SERVICE #GENERAL":10, "LOCATION #GENERAL":11}
    d = {"food": 0, "price": 1, "service": 2, "ambience": 3, "anecdotes/miscellaneous": 4}
    return d[cate]

def text2id(textfile, idfile, wdictfile):
    lines = map(lambda x: x.strip().split('\t'), open(wdictfile).readlines())
    wd = dict([(line[0], line[1]) for line in lines])
    text = map(lambda x: x.strip().split('<sss>'), open(textfile).readlines())
    reviews = map(lambda x: x[0].strip().split(' '), text)
    targets = map(lambda x: x[1].strip().split(' '), text)
    # category = map(lambda x: x[2].strip(), text)
    # cateid = map(lambda x: cate2id(x[1].strip()), text)
    sentimentids = map(lambda x: sentiment2id(x[2].strip()), text)
    # reviewids = map(lambda sentence: map(lambda word: wd[word], filter(lambda w: w != '', sentence)), reviews)
    reviewids = map(lambda sentence: map(lambda word: wd[word], sentence), reviews)
    targetids = map(lambda sentence: map(lambda word: wd[word], sentence), targets)
    tup = zip(reviewids, targetids, sentimentids)
    f = open(idfile, 'w')
    for i in range(len(tup)):
        re = tup[i][0]
        restr = ''
        for r in re:
            restr += r + ' '
        f.write(restr.strip() + '\t')
        tar = tup[i][1]
        tarstr = ''
        for t in tar:
            tarstr += t + ' '
        f.write(tarstr.strip() + '\t')
        senti = tup[i][2]
        f.write(str(senti) + '\n')
        # cate = tup[i][1]
        # senti = tup[i][2]
        # f.write(str(cate) + '\t' + str(senti) + '\n')
    f.close()

def text2idTrip(textfile, idfile, wdictfile):
    lines = map(lambda x: x.strip().split('\t'), open(wdictfile).readlines())
    wd = dict([(line[0], line[1]) for line in lines])
    f = open(idfile, 'w')
    for line in open(textfile, 'r'):
        review = line.strip().split('\t\t')
        f.write(review[0] + '\t\t')
        sentences = review[1].split('<ssssss>')
        for sen in sentences:
            words = sen.split(' ')
            for w in words:
                try:
                    id = wd[w]
                    f.write(id + ' ')
                except:
                    continue
        f.write('\n')
    f.close()

def analysisXML(XMLfile, trainfile, devfile):
    tree = ET.parse(XMLfile)
    root = tree.getroot()
    posi = 0
    nega = 0
    neu = 0
    ft = open(trainfile, 'w')
    fd = open(devfile, 'w')
    for i in range(len(root)):
            sentence = root[i]
            # for sentence in child[0]:
            if len(sentence) > 1 and sentence[1].tag == 'aspectTerms':
                review = sentence[0].text.lower()
                for op in sentence[1]:
                    tar = op.attrib['term'].lower()
                    # cate = op.attrib['category']
                    po = op.attrib['polarity']
                    # if po == 'positive':
                    #     posi += 1
                    # if po == 'negative':
                    #     nega += 1
                    # if po == 'neutral':
                    #     neu += 1
                    print review
                    if i % 6 == 1:
                        if po == 'positive':
                            posi += 1
                        if po == 'negative':
                            nega += 1
                        if po == 'neutral':
                            neu += 1
                        fd.write(review + '\t' + '<sss>' + '\t' + tar + '\t' + '<sss>' + '\t' + po + '\n')
                    else:
                        ft.write(review + '\t' + '<sss>' + '\t' + tar + '\t' + '<sss>' + '\t' + po + '\n')
                    # ft.write(
                        # review + '\t' + '<sss>' + '\t' + tar + '\t' + '<sss>' + '\t' + po + '\n')
    all = posi + nega + neu
    print str(posi) + '  ' + str(nega) + '  ' + str(neu) + '  ' + str(all)

def getPositionIndex(review, target):
    rlen = len(review)
    tlen = len(target)
    indexs = []
    for i in range(rlen - tlen + 1):
        flag = 1
        for j in range(tlen):
            if review[i + j] != target[j]:
                flag = 0
                break
        if flag == 1:
            for j in range(tlen):
               indexs.append(i+j)
    return indexs

def getindex(infile, outfile):
    txtlines = open(infile).readlines()
    lines = map(lambda x: x.strip().split('\t'), txtlines)
    reviews = map(lambda x: x[0].strip().split(' '), lines)
    targets = map(lambda x: x[1].strip().split(' '), lines)
    tup = zip(reviews, targets, txtlines)
    fw = open(outfile, 'w')
    for i in range(len(tup)):
        indexs = getPositionIndex(tup[i][0], tup[i][1])
        txt = tup[i][2]
        fw.write(txt.strip() + '\t')
        for j in range(len(indexs)):
            fw.write(str(indexs[j]) + ' ')
        fw.write('\n')
    fw.close()

def splitData(tfile, num):
    f = []
    for i in range(num):
        f.append(open(tfile+ '_' + str(i) + '.txt', 'w'))
    for line in open(tfile + '.txt', 'r'):
        strs = line.strip().split('\t')
        c = int(strs[1])
        f[c].write(line)
    for i in range(num):
        f[i].close()

def tripDataPre(trainidfile, trainfile, aspfile):
    ft = open(trainfile, 'w')
    fa = open(aspfile, 'w')
    for line in open(trainidfile, 'r'):
        strs = line.strip().split('\t\t')
        rates = strs[0].split(' ')
        review = strs[1]
        ratestr = ''
        for i in range(1, len(rates)):
            ra = rates[i]
            if ra == '-1':
                ratestr = ratestr + '0' + '\t'
            else:
                ratestr = ratestr + '1' + '\t'
        for i in range(1, len(rates)):
            ra = rates[i]
            if ra != '-1':
                ft.write(review + '\t' + str(i-1) + '\t' + str(int(ra)-1) + '\n')
                fa.write(ratestr + '\n')
    ft.close()
    fa.close()

base = '/home/zps/data/SemEval14/laptops/'
# base = '/home/zps/PycharmProjects/NeuralTextClassification/data/SemEval14/'
# wordFre([base + 'question_eval_set.txt', base + 'question_train_set.txt', base + 'topic_info.txt'], base + 'wordfre1.txt')
# parseXML(base + 'Laptops_Gold.xml', base + 'lap_test.txt')
# semEvalPre(base + 'restaurants_train_token.txt', base + 'res_train_term.txt', base + 'res_val_term.txt', base + 'res_train_asp.txt', base + 'res_val_asp.txt')
# countOOV(base + 'tripadvisor/t', base + 'SemEval14/data/glovewordlist.txt')
# sort_by_sen_length(base + 'lap_train_token.txt', base + 'lap_train_token_sorted.txt')
# embedding_pre('/home/zps/tools/glove/glove.42B.300d.txt', base + 'lap_train_token_sorted.txt', base + 'lap_test_token_sorted.txt',
#               base + 'lap_dev_token_sorted.txt', base + 'embedding.npy', base + 'wdict.txt')
# text2idTrip(base + 'train_sorted.txt', base + 'trainid.txt', base + 'wdict.txt')
text2id(base + 'lap_dev_token_sorted.txt', base + 'dev.txt', base + 'wdict.txt')
text2id(base + 'lap_test_token_sorted.txt', base + 'test.txt', base + 'wdict.txt')
text2id(base + 'lap_train_token_sorted.txt', base + 'train.txt', base + 'wdict.txt')
# tripDataPre(base + 'devid.txt', base + 'dev.txt', base + 'dev_a.txt')
getindex(base + 'train.txt', base + 'train_i.txt')
getindex(base + 'test.txt', base + 'test_i.txt')
getindex(base + 'dev.txt', base + 'dev_i.txt')
# analysisXML(base + 'Laptops_Train.xml', base + 'lap_train.txt', base + 'lap_dev.txt')
# splitData('/home/zps/PycharmProjects/NeuralTextClassification/data/Sem14ResAsp/dev', 5)
# base = '/media/zps/本地磁盘/03Paolo/Data/emnlp-2015-data/yelp-2013/'
# base = 'data/yelp13/'
# sourcetrain = base + 'train.txt'
# sourcetest = base + 'test.txt'
# sourcedev = base + 'dev.txt'
# filtertrain = base + 'yelp-2013-train-filter.txt'
# filtertest = base + 'yelp-2013-test-filter.txt'
# filterdev = base + 'yelp-2013-dev-filter.txt'

# sortedtrain = base + 'trainsorted.txt'
# sorteddev= base + 'devsorted.txt'
# sortedtest = base + 'testsorted.txt'
#
# usrlist = base + 'usrlist.txt'
# prdlist = base + 'prdlist.txt'
# wlist = base + 'wordlist.txt'
# usrDict = getDict(usrlist)
# prdDict = getDict(prdlist)
# wDict = getDict(wlist)

# corpus = base + 'w2v_traindev.txt'
# gene_corpus([sourcetrain, sourcedev], corpus)
# for (source, sort) in zip([sourcetrain, sourcedev, sourcetest], [sortedtrain, sorteddev, sortedtest]):
#     sort_by_sen_length(source, sort)

# pre(sorteddev, usrDict, prdDict, wDict, base + 'dev_x.txt', base + 'dev_y.txt', base + 'dev_u.txt', base + 'dev_p.txt')
# pre(sortedtrain, usrDict, prdDict, wDict, base + 'train_x.txt', base + 'train_y.txt', base + 'train_u.txt', base + 'train_p.txt')
# pre(sortedtest, usrDict, prdDict, wDict, base + 'test_x.txt', base + 'test_y.txt', base + 'test_u.txt', base + 'test_p.txt')

# np.array(pkl.load(base + 'embinit.save'))
# emb = np.load(base + 'embedding.npy')
# print emb
# arr = np.random.normal(loc=0.0, scale=0.1, size=(1,200))
# print arr
# emb[0] = arr
# print emb
# np.save(base + 'embedding_random.npy', emb)
# sourcelist = [sourcetrain, sourcetest, sourcedev]
# filterlist = [filtertrainsort, filtertestsort, filterdevsort]
# filterdotlist = [filterdottrainsort, filterdottestsort, filterdotdevsort]
# idset = [base + 'train_x.txt', base + 'test_x.txt', base + 'dev_x.txt']
# labelset = [base + 'train_y.txt', base + 'test_y.txt', base + 'dev_y.txt']
# w2v = base + 'yelp-2013-w2v_w10hsd200i15.vector'
# embedding = base + 'word_embedding.txt'

# wfd = word_fre(sourcelist)
# filter_low_fre(sourcelist, filterlist, wfd)
# filter_dot(filterlist, filterdotlist)
# sort_by_sen_length(filtertest, filtertestsort)
# sort_by_sen_length(filterdev, filterdevsort)
# sort_by_sen_length(filtertrain, filtertrainsort)
# gene_corpus([filtertrainsort, filterdev], corpus)
# dict = build_dict(w2v, embedding)
# word2id(filterdotlist, idset, labelset, dict)

# getindex(base + 'dev_old.txt', base + 'dev.txt')

# idsen = get_idsen_dict(base1 + 'senid.txt', base1 + 'input.tail.txt')
# get_sorted_xy(base2, 12, idsen, base2 + 'dev_x.txt', base2 + 'dev_y.txt')
#preRes(base1 + 'predict_result.txt', base1 + 'true_result.txt', base1 + 'res.txt')
# embedding_pre(base2 + 'input_w10hsd256i15.vector', base2 + 'word_embedding.txt')
# import tensorflow as tf
#
# with tf.Graph().as_default():
#     with tf.Session() as s:
#         # w = tf.Variable(
#         #     tf.random_uniform(shape=[5, 100], minval=0.01, maxval=0.01),
#         #     name='aspect_embedding', dtype=tf.float32, trainable=True)
#         w = tf.Variable(
#             tf.random_uniform(shape=[5, 100], minval=-0.01, maxval=0.01),
#             name='aspect_embedding', dtype=tf.float32, trainable=True)
#         s.run(tf.global_variables_initializer())
#         wa = s.run(w)
#         print wa
# from tensorflow.python import pywrap_tensorflow
# import os
# ckpt = tf.train.get_checkpoint_state("data/checkpoint_maxacc/")
#         # if ckpt and ckpt.model_checkpoint_path:
# checkpoint_path = ckpt.model_checkpoint_path
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
