/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.embeddings.wordvectors;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;

import java.util.*;

/**
 * Common word vector operations
 * @author Adam Gibson
 */
public class WordVectorsImpl<T extends SequenceElement> implements WordVectors {

    //number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
    @Getter protected int minWordFrequency = 5;
    @Getter protected WeightLookupTable<T> lookupTable;
    @Getter protected VocabCache<T> vocab;
    @Getter protected int layerSize = 100;
    @Getter protected transient ModelUtils<T> modelUtils = new BasicModelUtils<>();
    private boolean initDone = false;

    protected int numIterations = 1;
    protected int numEpochs = 1;
    protected double negative = 0;
    protected double sampling = 0;
    protected AtomicDouble learningRate = new AtomicDouble(0.025);
    protected double minLearningRate = 0.01;
    @Getter protected int window = 5;
    protected int batchSize;
    protected int learningRateDecayWords;
    protected boolean resetModel;
    protected boolean useAdeGrad;
    protected int workers = Runtime.getRuntime().availableProcessors();
    protected boolean trainSequenceVectors = false;
    protected boolean trainElementsVectors = true;
    protected long seed;
    protected boolean useUnknown = false;


    public final static String DEFAULT_UNK = "UNK";
    @Getter @Setter private String UNK = DEFAULT_UNK;

    @Getter protected List<String> stopWords = new ArrayList<>(); //StopWords.getStopWords();
    /**
     * Returns true if the model has this word in the vocab
     * @param word the word to test for
     * @return true if the model has the word in the vocab
     */
    public boolean hasWord(String word) {
        return vocab().indexOf(word) >= 0;
    }
    /**
     * Words nearest based on positive and negative words
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    public Collection<String> wordsNearestSum(Collection<String> positive,Collection<String> negative,int top) {
        return modelUtils.wordsNearestSum(positive, negative, top);
    }

    /**
     * Words nearest based on positive and negative words
     * * @param top the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearestSum(INDArray words,int top) {
        return modelUtils.wordsNearestSum(words, top);
    }

    /**
     * Words nearest based on positive and negative words
     * * @param top the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        return modelUtils.wordsNearest(words, top);
    }

    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    public Collection<String> wordsNearestSum(String word,int n) {
        return modelUtils.wordsNearestSum(word, n);
    }


    /**
     * Accuracy based on questions which are a space separated list of strings
    * where the first word is the query word, the next 2 words are negative,
    * and the last word is the predicted word to be nearest
    * @param questions the questions to ask
    * @return the accuracy based on these questions
    */
    public Map<String,Double> accuracy(List<String> questions) {
        return modelUtils.accuracy(questions);
    }

    @Override
    public int indexOf(String word) {
        return vocab().indexOf(word);
    }


    /**
     * Find all words with a similar characters
     * in the vocab
     * @param word the word to compare
     * @param accuracy the accuracy: 0 to 1
     * @return the list of words that are similar in the vocab
     */
    public List<String> similarWordsInVocabTo(String word,double accuracy) {
        return this.modelUtils.similarWordsInVocabTo(word, accuracy);
    }

    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    public double[] getWordVector(String word) {
        int i = vocab().indexOf(word);
        if(i < 0)
            return null;
        return lookupTable.vector(word).dup().data().asDouble();
    }

    /**
     * Returns the word vector divided by the norm2 of the array
     * @param word the word to get the matrix for
     * @return the looked up matrix
     */
    public INDArray getWordVectorMatrixNormalized(String word) {
        int i = vocab().indexOf(word);

        if(i < 0)
            return null;
        INDArray r =  lookupTable().vector(word);
        return r.div(Nd4j.getBlasWrapper().nrm2(r));
    }

    @Override
    public INDArray getWordVectorMatrix(String word) {
        return lookupTable().vector(word);
    }


    /**
     * Words nearest based on positive and negative words
     *
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        return modelUtils.wordsNearest(positive, negative, top);
    }

    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    public Collection<String> wordsNearest(String word,int n) {
       return modelUtils.wordsNearest(word, n);
    }


    /**
     * Returns similarity of two elements, provided by ModelUtils
     *
     * @param word the first word
     * @param word2 the second word
     * @return a normalized similarity (cosine similarity)
     */
    public double similarity(String word,String word2) {
        return modelUtils.similarity(word, word2);
    }

    @Override
    public VocabCache<T> vocab() {
        return vocab;
    }

    @Override
    public WeightLookupTable lookupTable() {
        return lookupTable;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void setModelUtils(@NonNull ModelUtils modelUtils) {
        if (lookupTable != null) {
            modelUtils.init(lookupTable);
            this.modelUtils = modelUtils;
        }
    }

    public void setLookupTable(@NonNull WeightLookupTable lookupTable) {
        this.lookupTable = lookupTable;
        if (modelUtils == null) this.modelUtils = new BasicModelUtils<T>();

        this.modelUtils.init(lookupTable);
    }

    public void setVocab(VocabCache vocab) {
        this.vocab = vocab;
    }

    protected void update() {
        update(EnvironmentUtils.buildEnvironment(), Event.STANDALONE);
    }

    protected void update(Environment env, Event event) {
        if (!initDone) {
            initDone = true;

            Heartbeat heartbeat = Heartbeat.getInstance();
            Task task = new Task();
            task.setNumFeatures(layerSize);
            if (vocab != null) task.setNumSamples(vocab.numWords());
            task.setNetworkType(Task.NetworkType.DenseNetwork);
            task.setArchitectureType(Task.ArchitectureType.WORDVECTORS);

            heartbeat.reportEvent(event, env, task);
        }
    }
}
