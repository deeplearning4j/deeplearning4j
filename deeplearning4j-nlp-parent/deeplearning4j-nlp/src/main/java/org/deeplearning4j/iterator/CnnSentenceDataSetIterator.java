package org.deeplearning4j.iterator;

import lombok.AllArgsConstructor;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 27/01/2017.
 */
@AllArgsConstructor
public class CnnSentenceDataSetIterator implements DataSetIterator {

    public enum UnknownWordHandling {RemoveWord, UseUnknownVector}

    private static final String UNKNOWN_WORD_SENTINAL = "UNKNOWN_WORD_SENTINAL";

    private LabelledSentenceProvider sentenceProvider = null;
    private WordVectors wordVectors;
    private TokenizerFactory tokenizerFactory;
    private UnknownWordHandling unknownWordHandling;
    private boolean useNormalizedWordVectors;
    private int minibatchSize;
    private int maxSentenceLength;
    private boolean sentencesAlongHeight;
    private DataSetPreProcessor dataSetPreProcessor;

    private int wordVectorSize;
    private int numClasses;
    private Map<String, Integer> labelClassMap;
    private INDArray unknown;

    private int cursor = 0;

    private CnnSentenceDataSetIterator(Builder builder) {
        this.sentenceProvider = builder.sentenceProvider;
        this.tokenizerFactory = builder.tokenizerFactory;
        this.unknownWordHandling = builder.unknownWordHandling;
        this.useNormalizedWordVectors = builder.useNormalizedWordVectors;
        this.minibatchSize = builder.minibatchSize;
        this.maxSentenceLength = builder.maxSentenceLength;
        this.sentencesAlongHeight = builder.sentencesAlongHeight;
        this.dataSetPreProcessor = builder.dataSetPreProcessor;


        this.numClasses = this.sentenceProvider.numLabelClasses();
        this.labelClassMap = new HashMap<>();
        int count = 0;
        for (String s : this.sentenceProvider.allLabels()) {
            this.labelClassMap.put(s, count++);
        }
        if(unknownWordHandling == UnknownWordHandling.UseUnknownVector){
            if(useNormalizedWordVectors){
                wordVectors.getWordVectorMatrixNormalized(wordVectors.getUNK());
            } else {
                wordVectors.getWordVectorMatrix(wordVectors.getUNK());
            }
        }
    }

    @Override
    public boolean hasNext() {
        if (sentenceProvider == null) {
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
        if (sentenceProvider == null) {
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }


        List<Pair<List<String>, String>> tokenizedSentences = new ArrayList<>(num);
        int maxLength = -1;
        int minLength = Integer.MAX_VALUE; //Track to we know if we can skip mask creation for "all same length" case
        for (int i = 0; i < num && sentenceProvider.hasNext(); i++) {
            Pair<String, String> p = sentenceProvider.nextSentence();
            List<String> tokens = new ArrayList<>();
            Tokenizer t = tokenizerFactory.create(p.getFirst());
            while (t.hasMoreTokens()) {
                String token = t.nextToken();
                if (!wordVectors.hasWord(token)) {
                    switch (unknownWordHandling) {
                        case RemoveWord:
                            continue;
                        case UseUnknownVector:
                            token = UNKNOWN_WORD_SENTINAL;
                    }
                }
                tokens.add(token);
            }

            maxLength = Math.max(maxLength, tokens.size());
            tokenizedSentences.add(new Pair<>(tokens, p.getSecond()));
        }

        if (maxSentenceLength > 0 && maxLength > maxSentenceLength) {
            maxLength = maxSentenceLength;
        }

        int currMinibatchSize = tokenizedSentences.size();
        INDArray labels = Nd4j.create(currMinibatchSize, numClasses);
        for (int i = 0; i < tokenizedSentences.size(); i++) {
            String labelStr = tokenizedSentences.get(i).getSecond();
            if (!labelClassMap.containsKey(labelStr)) {
                throw new IllegalStateException("Got label \"" + labelStr + "\" that is not present in list of LabelledSentenceProvider labels");
            }

            int labelIdx = labelClassMap.get(labelStr);

            labels.putScalar(i, labelIdx);
        }

        int[] featuresShape = new int[4];
        featuresShape[0] = currMinibatchSize;
        featuresShape[1] = 1;
        if (sentencesAlongHeight) {
            featuresShape[2] = maxSentenceLength;
            featuresShape[3] = wordVectorSize;
        } else {
            featuresShape[2] = wordVectorSize;
            featuresShape[3] = maxSentenceLength;
        }

        INDArray features = Nd4j.create(featuresShape);
        for (int i = 0; i < currMinibatchSize; i++) {
            List<String> currSentence = tokenizedSentences.get(i).getFirst();

            for (int j = 0; j < currSentence.size() && j <maxSentenceLength; j++) {
                String word = currSentence.get(j);
                INDArray vector;
                if(unknownWordHandling == UnknownWordHandling.UseUnknownVector && word == UNKNOWN_WORD_SENTINAL){    //Yes, this *should* be using == for the sentinal String here
                    vector = unknown;
                } else {
                    if(useNormalizedWordVectors){
                        vector = wordVectors.getWordVectorMatrixNormalized(word);
                    } else {
                        vector = wordVectors.getWordVectorMatrix(word);
                    }
                }

                INDArrayIndex[] indices = new INDArrayIndex[4];
                //TODO REUSE
                indices[0] = NDArrayIndex.point(i);
                indices[1] = NDArrayIndex.point(0);
                if(sentencesAlongHeight){
                    indices[2] = NDArrayIndex.point(j);
                    indices[3] = NDArrayIndex.all();
                } else {
                    indices[2] = NDArrayIndex.all();
                    indices[3] = NDArrayIndex.point(j);
                }

                features.put(indices, vector);
            }
        }

        INDArray featuresMask = null;
        if (minLength != maxLength) {
            featuresMask = Nd4j.create(currMinibatchSize, maxLength);

            for()
        }

        DataSet ds = new DataSet(features, labels, featuresMask, null);

        if (dataSetPreProcessor != null) {
            dataSetPreProcessor.preProcess(ds);
        }

        cursor += ds.numExamples();
        return ds;
    }

    @Override
    public int totalExamples() {
        return 0;
    }

    @Override
    public int inputColumns() {
        return wordVectorSize;
    }

    @Override
    public int totalOutcomes() {
        return numClasses;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
        sentenceProvider.reset();
    }

    @Override
    public int batch() {
        return minibatchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        //TODO - not always knowable?
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return dataSetPreProcessor;
    }

    @Override
    public List<String> getLabels() {
        return sentenceProvider.allLabels();
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
        private boolean useNormalizedWordVectors = true;
        private int maxSentenceLength = -1;
        private int minibatchSize = 32;
        private boolean sentencesAlongHeight = true;
        private DataSetPreProcessor dataSetPreProcessor;

        public Builder labelledSentenceProvider(LabelledSentenceProvider labelledSentenceProvider) {
            this.sentenceProvider = labelledSentenceProvider;
            return this;
        }

        public Builder wordVectors(WordVectors wordVectors) {
            this.wordVectors = wordVectors;
            return this;
        }

        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder unknownWordHandling(UnknownWordHandling unknownWordHandling) {
            this.unknownWordHandling = unknownWordHandling;
            return this;
        }

        public Builder minibatchSize(int minibatchSize) {
            this.minibatchSize = minibatchSize;
            return this;
        }

        public Builder useNormalizedWordVectors(boolean useNormalizedWordVectors){
            this.useNormalizedWordVectors = useNormalizedWordVectors;
            return this;
        }

        public Builder maxSentenceLength(int maxSentenceLength) {
            this.maxSentenceLength = maxSentenceLength;
            return this;
        }

        public Builder sentencesAlongHeight(boolean sentencesAlongHeight) {
            this.sentencesAlongHeight = sentencesAlongHeight;
            return this;
        }

        public Builder dataSetPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
            this.dataSetPreProcessor = dataSetPreProcessor;
            return this;
        }

        public CnnSentenceDataSetIterator build() {
            return new CnnSentenceDataSetIterator(this);
        }

    }
}
