package org.deeplearning4j.bagofwords.vectorizer;

import org.apache.commons.io.IOUtils;
import org.apache.uima.util.FileUtils;
import org.deeplearning4j.berkeley.Counter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.Index;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Bag of words vectorizer.
 * Transforms a document in to a bag of words
 * @author Adam Gibson
 *
 */
public class BagOfWordsVectorizer implements TextVectorizer {

    private LabelAwareSentenceIterator sentenceIter;
    private TokenizerFactory tokenizerFactory;
    private Counter<String> wordCounts;
    private Index vocab;
    private int vocabSize;
    private List<String> stopWords;
    private List<String> labels;
    /**
     * Converts a document in to a bag of words
     * @param sentenceIterator the sentence iterator to use
     * This handles segmenting the document in to
     * whole segments
     * @param tokenizerFactory the tokenizer to use
     * @param labels the possible labels for each document
     * @param vocabSize the max size of vocab
     */
    public BagOfWordsVectorizer(LabelAwareSentenceIterator sentenceIterator, TokenizerFactory tokenizerFactory,  List<String> labels, int vocabSize) {
        this.sentenceIter = sentenceIterator;
        this.tokenizerFactory = tokenizerFactory;
        this.vocab = new Index();
        this.labels = labels;
        this.vocabSize = vocabSize;
        wordCounts = new Counter<>();
        stopWords = StopWords.getStopWords();
    }


    /**
     * Converts a document in to a bag of words
     * @param sentenceIterator the sentence iterator to use
     * This handles segmenting the document in to
     * whole segments
     * @param tokenizerFactory the tokenizer to use
     * @param labels the possible labels for each document
     */
    public BagOfWordsVectorizer(LabelAwareSentenceIterator sentenceIterator, TokenizerFactory tokenizerFactory,  List<String> labels) {
       this(sentenceIterator,tokenizerFactory,labels,-1);
    }


    /**
     * The vocab sorted in descending order
     *
     * @return the vocab sorted in descending order
     */
    @Override
    public Index vocab() {
        return vocab;
    }

    /**
     * Text coming from an input stream considered as one document
     *
     * @param is    the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(InputStream is, String label) {
        try {
            String inputString = IOUtils.toString(is);
            return vectorize(inputString,label);

        }catch(Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Vectorizes the passed in text treating it as one document
     *
     * @param text  the text to vectorize
     * @param label the label of the text
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(String text, String label) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray input = Nd4j.create(1,vocab.size());
        for(int i = 0; i < tokens.size(); i++) {
            int idx = vocab.indexOf(tokens.get(i));
            if(vocab.indexOf(tokens.get(i)) >= 0)
                input.putScalar(idx,wordCounts.getCount(tokens.get(i)));
        }

        INDArray labelMatrix = FeatureUtil.toOutcomeVector(labels.indexOf(label), labels.size());
        return new DataSet(input,labelMatrix);
    }

    private void process() {
        while(sentenceIter.hasNext()) {
            Tokenizer tokenizer = tokenizerFactory.create(sentenceIter.nextSentence());
            List<String> tokens = tokenizer.getTokens();
            for(String token  : tokens)
                if(!stopWords.contains(token)) {
                    wordCounts.incrementCount(token,1.0);
                   if(vocab.indexOf(token) < 0)
                       vocab.add(token);
                }
        }
    }


    /**
     * @param input the text to vectorize
     * @param label the label of the text
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(File input, String label) {
        try {
            String text = FileUtils.file2String(input);
            return vectorize(text, label);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public DataSet vectorize() {
        process();
        sentenceIter.reset();
        List<DataSet> ret = new ArrayList<>();
        while(sentenceIter.hasNext()) {
            ret.add(vectorize(sentenceIter.nextSentence(),sentenceIter.currentLabel()));
        }
        return DataSet.merge(ret);
    }

    /**
     * Transforms the matrix
     *
     * @param text
     * @return
     */
    @Override
    public INDArray transform(String text) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray input = Nd4j.create(1, vocab.size());
        for(int i = 0; i < tokens.size(); i++) {
            int idx = vocab.indexOf(tokens.get(i));
            if(vocab.indexOf(tokens.get(i)) >= 0)
                input.putScalar(idx, wordCounts.getCount(tokens.get(i)));
        }
        return input;
    }
}
