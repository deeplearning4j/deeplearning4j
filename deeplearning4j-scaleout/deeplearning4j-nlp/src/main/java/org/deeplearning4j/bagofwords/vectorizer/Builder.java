package org.deeplearning4j.bagofwords.vectorizer;

import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.invertedindex.DefaultInvertedIndex;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.lang.reflect.Constructor;
import java.util.List;

public abstract class Builder {
    protected VocabCache cache = new InMemoryLookupCache(100);
    protected TokenizerFactory tokenizerFactory;
    protected List<String> stopWords = StopWords.getStopWords();
    protected int layerSize = 1;
    protected int minWordFrequency = 5;
    protected DocumentIterator docIter;
    protected SentenceIterator sentenceIterator;
    protected List<String> labels;
    protected InvertedIndex index;


    public Builder index(InvertedIndex index){
        this.index = index;
        return this;
    }

    public Builder labels(List<String> labels) {
        this.labels = labels;
        return this;
    }


    public Builder cache(VocabCache cache) {
        this.cache = cache;
        return this;
    }

    public Builder tokenize(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
        return this;
    }

    public Builder stopWords(List<String> stopWords) {
        this.stopWords = stopWords;
        return this;
    }

    public Builder layerSize(int layerSize) {
        this.layerSize = layerSize;
        return this;
    }

    public Builder minWords(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
        return this;
    }

    public Builder iterate(DocumentIterator docIter) {
        this.docIter = docIter;
        return this;
    }

    public Builder iterate(SentenceIterator sentenceIterator) {
        this.sentenceIterator = sentenceIterator;
        return this;
    }



    public abstract TextVectorizer build();
}