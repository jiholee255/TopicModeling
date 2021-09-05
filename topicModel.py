import tomotopy as tp
import pymysql
import sys
import re
import pandas as pd

con = pymysql.connect(
    host="", port=, db="", user="", password=""
)
curs = con.cursor()

sql = """
SELECT p_index, group_concat(kwd) from new_tpf
WHERE use_f = "Y" AND `type`='function'
GROUP BY p_index;
"""

rev_sql = """
SELECT p_index,paragraph FROM paragraph
WHERE sentiment_score<=0
GROUP BY p_index;
"""
curs.execute(rev_sql)
rows = curs.fetchall()


def set_lda_topicNum(input_data):
    stop_word_lst = []
    with open("./stop_words_english.txt", encoding="UTF8") as stw:
        line = stw.readlines()
        for l in line:
            stop_word_lst.append(l.strip())
    ret_lst = []
    for k in range(1, 51):
        tmp_lst = []
        mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=300, k=k)
        for n, line in enumerate(input_data):
            re_str = re.sub(r"[\"”“:!',*)@#%(&$_?.’^]", "", line[1])
            ch = [k for k in re_str.strip().split(" ") if len(k) > 3 and k not in stop_word_lst]
            if len(ch) > 0:
                mdl.add_doc(ch)
        mdl.burn_in = 100
        mdl.train(10)
        for i in range(0, 1000, 10):
            mdl.train(0)
        tmp_lst.append("Topic:" + str(k))
        tmp_lst.append(mdl.perplexity)
        tmp_lst.append(mdl.ll_per_word)
        ret_lst.append(tmp_lst)
    pd.DataFrame(ret_lst).to_csv("topic_num.csv", index=False)


def lda_example(input_data, topic_num):
    stop_word_lst = []
    with open("./stop_words_english.txt", encoding="UTF8") as stw:
        line = stw.readlines()
        for l in line:
            stop_word_lst.append(l.strip())
    mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=500, k=topic_num)
    for n, line in enumerate(input_data):
        re_str = re.sub(r"[\"”“:!',*)@#%(&$_?.’^]", "", line[1])
        ch = [k for k in re_str.strip().split(" ") if len(k) > 3 and k not in stop_word_lst]
        if len(ch) > 0:
            mdl.add_doc(ch)
    mdl.burn_in = 100
    mdl.train(10)
    # print(
    #     "Num docs:",
    #     len(mdl.docs),
    #     ", Vocab size:",
    #     len(mdl.used_vocabs),
    #     ", Num words:",
    #     mdl.num_words,
    # )
    # print("Removed top words:", mdl.removed_top_words)
    # print("Training...", file=sys.stderr, flush=True)
    for i in range(0, 1000, 10):
        mdl.train(10)
        # print("Iteration: {}\tLog-likelihood: {}".format(i, mdl.ll_per_word))

    mdl.summary()
    #     print("Saving...", file=sys.stderr, flush=True)
    #     mdl.save(save_path, True)
    ret_lst = []
    for k in range(mdl.k):
        tmp_lst = []
        tmp_lst.append("Topic #{}".format(k))
        for word, prob in mdl.get_topic_words(k):
            tmp_lst.append(word)
            tmp_lst.append(prob)
        ret_lst.append(tmp_lst)
    pd.DataFrame(ret_lst).to_csv("./TopicWords.csv", index=False)


lda_example(rows, 33)
# set_lda_topicNum(rows)
