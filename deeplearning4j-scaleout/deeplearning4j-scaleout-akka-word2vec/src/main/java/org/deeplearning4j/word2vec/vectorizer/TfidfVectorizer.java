package org.deeplearning4j.word2vec.vectorizer;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.SetUtils;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.DefaultTokenizerFactory;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.viterbi.Index;

import java.util.List;
import java.util.Collection;

/**
 * Turns a set of documents in to a tfidf bag of words
 * @author Adam Gibson
 */
public class TfidfVectorizer implements Vectorizer {

    private int minDf;
    private Index vocab = new Index();
    private SentenceIterator sentenceIterator;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    private List<String> labels;
    private Counter<String> tf = new Counter<>();
    private Counter<String> idf = new Counter<>();
    private int numDocs = 0;

    /**
     *
     * @param sentenceIterator the document iterator
     * @param tokenizerFactory the tokenizer for individual tokens
     * @param labels the possible labels
     */
    public TfidfVectorizer(SentenceIterator sentenceIterator,TokenizerFactory tokenizerFactory,List<String> labels) {
        this.sentenceIterator = sentenceIterator;
        this.tokenizerFactory = tokenizerFactory;
        this.labels = labels;


    }

    private Collection<String> allWords() {
        return SetUtils.union(tf.keySet(),idf.keySet());
    }


    private Counter<String> tfIdfWeights(int top) {
        Counter<String> ret = new Counter<String>();
        for(String word  : allWords())
            ret.setMinCount(word,tfidfWord(word));
        ret.keepTopNKeys(top);
        return ret;
    }


    private Counter<String> tfIdfWeights() {
        Counter<String> ret = new Counter<String>();
        for(String word  : allWords())
            ret.setMinCount(word,tfidfWord(word));
        return ret;
    }

    private void initIndexFromTfIdf() {
        Counter<String> tfidf = tfIdfWeights();
        for(String key : tfidf.keySet())
            this.vocab.add(key);
    }



    private double tfidfWord(String word) {
        return MathUtils.tfidf(tfForWord(word),idfForWord(word));
    }


    private double tfForWord(String word) {
        return MathUtils.tf((int) tf.getCount(word));
    }

    private double idfForWord(String word) {
        return MathUtils.idf(numDocs,idf.getCount(word));
    }

    private void process() {
        while(sentenceIterator.hasNext()) {
            Tokenizer tokenizer = tokenizerFactory.create(sentenceIterator.nextSentence());
            Counter<String> runningTotal = new Counter<>();
            Counter<String> documentOccurrences = new Counter<>();
            numDocs++;
            for(String token : tokenizer.getTokens()) {
                runningTotal.incrementCount(token,1.0);
                //idf
                if(!documentOccurrences.containsKey(token))
                    documentOccurrences.setMinCount(token,1.0);
            }

            idf.incrementAll(documentOccurrences);
            tf.incrementAll(runningTotal);
        }
    }


    @Override
    public DataSet vectorize() {
        process();
        initIndexFromTfIdf();
        
        Counter<String> tfIdfWeights = tfIdfWeights();


        return null;
    }
}
