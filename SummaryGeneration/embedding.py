import pickle
from time import time

import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

def main():
    # load dataset
    df = pd.read_csv("./data/dfPreProcessed.csv")
    # extract training data, the abstract is our source text and title our summary

    df_train = df[["abstract", "title"]].astype(str)   # abstract predicts title

    df_train['abstract_word_tokens'] = df_train['abstract'].apply(word_tokenize)
    df_train['title_word_tokens'] = df_train['title'].apply(word_tokenize)

    doc = [[df_train.loc[i, 'abstract_word_tokens'], df_train.loc[i, 'title_word_tokens']] for i in range(len(df_train))]

    doc_source = [doc[i][0] for i in range(len(doc))]
    doc_summary = [doc[i][1] for i in range(len(doc))]

    print(doc_source[0], doc_summary[0])

    model = Word2Vec(min_count=20,
                     window=2,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=4)

    t = time()

    total_text = doc_source
    total_text.extend(doc_summary)

    model.build_vocab(total_text, progress_per=10000)
    print(len(model.wv.vocab.keys()))

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    print(model.corpus_count)

    t = time()
    # model.train([["hello", "world"]], total_examples=1, epochs=30, report_delay=1)
    model.train(total_text, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    # save model and data
    model.save("./data/word2vec_sum.model")

    with open('./data/doc_source.p', 'wb') as file_pi:
        pickle.dump(doc_source, file_pi)

    with open('./data/doc_summary.p', 'wb') as file_pi:
        pickle.dump(doc_source, file_pi)


if __name__ == '__main__':
    main()
#