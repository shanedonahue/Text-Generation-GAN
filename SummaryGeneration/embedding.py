import pickle
from time import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import gensim.downloader as gensim_api
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

def main():
    # load dataset
    df = pd.read_csv("./data/dfPreProcessed.csv")
    # extract training data, the abstract is our source text and title our summary
    df_train = df[["abstract", "title"]].astype(str)   # abstract predicts title

    doc_source = df_train['abstract'].tolist()
    doc_summary = df_train['title'].tolist()

    with open('./data/doc_source.p', 'wb') as source_file:
        pickle.dump(doc_source, source_file)

    with open('./data/doc_summary.p', 'wb') as summary_file:
        pickle.dump(doc_summary, summary_file)

    texts = doc_source.copy()
    texts.extend(doc_summary)
    print(texts[0:10])
    sentences = []
    for i in range(len(texts)):
        sentences.append(sent_tokenize(texts[i]))

    sents = []
    for sentence_list in sentences:
        for sentence in sentence_list:
            sents.append(word_tokenize(sentence))

    print(sents[0:3])

    model = Word2Vec(vector_size=50, min_count=1)
    model.build_vocab(sents)
    total_examples = model.corpus_count
    print(total_examples)
    # Save the vocab of your dataset
    vocab = list(model.wv.key_to_index.keys())
    print(vocab[0])

    pretrained_path = "./glove/glove.6B.50d.word2vec.txt"
    # glove2word2vec("./glove/glove.6B.50d.txt", pretrained_path)
    glove_model_wv = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)

    print(glove_model_wv.most_similar("cats"))

    # Add the pre-trained model vocabulary
    model.build_vocab([list(glove_model_wv.key_to_index.keys())], update=True)

    # Load the pre-trained models embeddings
    # note: if a word doesn't exist in the pre-trained vocabulary then it is left as is in the original model
    model.wv.vectors_lockf = np.ones(len(model.wv))
    model.wv.intersect_word2vec_format(pretrained_path, binary=False, lockf=1.0)

    model.train(sents, total_examples=total_examples, epochs=model.epochs)

    print(model.wv.most_similar("cats"))

    # save model and data
    model.save("./data/glove_word2vec_50d.model")




    # doc = [[df_train.loc[i, 'abstract_word_tokens'], df_train.loc[i, 'title_word_tokens']] for i in range(len(df_train))]
    #
    # doc_source = [doc[i][0] for i in range(len(doc))]
    # doc_summary = [doc[i][1] for i in range(len(doc))]
    #
    # print(doc_source[0], doc_summary[0])
    #
    # model = Word2Vec(min_count=20,
    #                  window=2,
    #                  sample=6e-5,
    #                  alpha=0.03,
    #                  min_alpha=0.0007,
    #                  negative=20,
    #                  workers=4)
    #
    # t = time()
    #
    # total_text = doc_source
    # total_text.extend(doc_summary)
    #
    # model.build_vocab(total_text, progress_per=10000)
    # print(len(model.wv.vocab.keys()))
    #
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    # print(model.corpus_count)
    #
    # t = time()
    # # model.train([["hello", "world"]], total_examples=1, epochs=30, report_delay=1)
    # model.train(total_text, total_examples=model.corpus_count, epochs=30, report_delay=1)
    # print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    #





if __name__ == '__main__':
    main()
#