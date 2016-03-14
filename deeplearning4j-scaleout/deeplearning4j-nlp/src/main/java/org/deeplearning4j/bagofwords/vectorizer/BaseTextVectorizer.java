package org.deeplearning4j.bagofwords.vectorizer;

import lombok.Getter;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseTextVectorizer implements TextVectorizer {
    protected transient TokenizerFactory tokenizerFactory;
    protected transient LabelAwareIterator iterator;
    protected int minWordFrequency;
    @Getter protected VocabCache<VocabWord> vocabCache;
    protected LabelsSource labelsSource;
    protected List<String> stopWords = new ArrayList<>();
    @Getter protected transient InvertedIndex<VocabWord> index;

    protected LabelsSource getLabelsSource() {
        return labelsSource;
    }

    public void buildVocab() {
        if (vocabCache == null) vocabCache = new AbstractCache.Builder<VocabWord>().build();


        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(this.iterator)
                .tokenizerFactory(tokenizerFactory)
                .build();

        AbstractSequenceIterator<VocabWord> iterator = new AbstractSequenceIterator.Builder<>(transformer)
                .build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(iterator, minWordFrequency)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);
    }

    @Override
    public void fit() {
        buildVocab();
    }

    /**
     * Returns the number of words encountered so far
     *
     * @return the number of words encountered so far
     */
    @Override
    public long numWordsEncountered() {
        return vocabCache.totalWordOccurrences();
    }
}
