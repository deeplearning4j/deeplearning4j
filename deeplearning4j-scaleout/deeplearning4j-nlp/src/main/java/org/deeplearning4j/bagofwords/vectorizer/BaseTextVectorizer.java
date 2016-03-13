package org.deeplearning4j.bagofwords.vectorizer;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseTextVectorizer implements TextVectorizer {
    protected TokenizerFactory tokenizerFactory;
    protected SentenceIterator iterator;
    protected int minWordFrequency;
    protected VocabCache<VocabWord> vocabCache;
    protected LabelsSource labelsSource;

    public void buildVocab() {

    }
}
