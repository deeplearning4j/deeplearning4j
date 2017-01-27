package org.deeplearning4j.iterator;

import lombok.AllArgsConstructor;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.DefaultTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 27/01/2017.
 */
@AllArgsConstructor
public class CnnSentenceDataSetIterator implements DataSetIterator {

    public enum UnknownWordHandling {RemoveWord, UseUnknownVector};

    private static final String UNKNOWN_WORD_SENTINAL = "UNKNOWN_WORD_SENTINAL";

    private LabelledSentenceProvider sentenceProvider = null;
    private WordVectors wordVectors;
    private TokenizerFactory tokenizerFactory;
    private UnknownWordHandling unknownWordHandling;
    private int minibatchSize;
    private int maxSentenceLength;


    private CnnSentenceDataSetIterator(Builder builder){
        this.sentenceProvider = builder.sentenceProvider;
        this.tokenizerFactory = builder.tokenizerFactory;
        this.unknownWordHandling = builder.unknownWordHandling;
        this.minibatchSize = builder.minibatchSize;
        this.maxSentenceLength = builder.maxSentenceLength;
    }

    @Override
    public boolean hasNext() {
        if(sentenceProvider == null){
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }
        return sentenceProvider.hasNext();
    }

    @Override
    public DataSet next() {
        return next(minibatchSize);
    }

    @Override
    public DataSet next(int num) {
        if(sentenceProvider == null){
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }


        List<Pair<String,String>> tokenizedSentences = new ArrayList<>(num);
        int maxLength = -1;
        for( int i=0; i<num && sentenceProvider.hasNext(); i++ ){
            Pair<String,String> p = sentenceProvider.nextSentence();
            List<String> tokens = new ArrayList<>();
            Tokenizer t = tokenizerFactory.create(p.getFirst());
            while(t.hasMoreTokens()){
                String token = t.nextToken();
                if(!wordVectors.hasWord(token)){
                    switch (unknownWordHandling){
                        case RemoveWord:
                            continue;
                        case UseUnknownVector:
                            token = UNKNOWN_WORD_SENTINAL;
                    }
                }
            }

            maxLength = Math.max(maxLength, tokens.size());
            tokenizedSentences.add(p);
        }


        return null;
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return 0;
    }

    @Override
    public int cursor() {
        return 0;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    public static class Builder {

        private LabelledSentenceProvider sentenceProvider = null;
        private WordVectors wordVectors;
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private UnknownWordHandling unknownWordHandling = UnknownWordHandling.RemoveWord;
        private int maxSentenceLength = -1;
        private int minibatchSize = 32;
        private boolean sentencesAlongHeight = true;

        public Builder labelledSentenceProvider(LabelledSentenceProvider labelledSentenceProvider){
            this.sentenceProvider = labelledSentenceProvider;
            return this;
        }

        public Builder wordVectors(WordVectors wordVectors){
            this.wordVectors = wordVectors;
            return this;
        }

        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory){
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder unknownWordHandling(UnknownWordHandling unknownWordHandling){
            this.unknownWordHandling = unknownWordHandling;
            return this;
        }

        public Builder minibatchSize(int minibatchSize){
            this.minibatchSize = minibatchSize;
            return this;
        }

        public Builder maxSentenceLength(int maxSentenceLength){
            this.maxSentenceLength = maxSentenceLength;
            return this;
        }

        public Builder sentencesAlongHeight(boolean sentencesAlongHeight){
            this.sentencesAlongHeight = sentencesAlongHeight;
            return this;
        }

        public CnnSentenceDataSetIterator build(){
            return new CnnSentenceDataSetIterator(this);
        }

    }
}
