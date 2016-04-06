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

package org.deeplearning4j.models.rntn;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.util.MultiDimensionalMap;
import org.deeplearning4j.util.MultiDimensionalSet;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;
import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;

import com.google.common.util.concurrent.AtomicDouble;
/**
 * Recursive Neural Tensor Network by Socher et. al
 *
 * This is a modified implementation of the sentiment analysis RNTN
 * from Stanford that is intended to work with more general purpose inputs (scene detection with images,
 * labeling
 * series of sentences, among others)
 *
 * This implementation will also be faster in terms of
 * parallelization as well as integration with
 * native matrices/GPUs
 *
 * @author Adam Gibson
 *
 */
@Deprecated
public class RNTN implements Layer {


    protected NeuralNetConfiguration conf;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected double value = 0;
    private int numOuts = 3;
    //must be same size as word vectors
    private int numHidden = 25;
    private Random rng;
    private boolean useDoubleTensors = true;
    private boolean combineClassification = true;
    private boolean simplifiedModel = true;
    private boolean randomFeatureVectors = true;
    private double scalingForInit = 1.0f;
    private boolean lowerCasefeatureNames;
    protected String activationFunction = "tanh";
    protected String outputActivation = "softmax";
    protected AdaGrad paramAdaGrad;
    protected int numParameters = -1;
    /** Regularization cost for the applyTransformToOrigin matrix  */
    private double regTransformMatrix = 0.001f;

    /** Regularization cost for the classification matrices */
    private double regClassification = 0.0001f;

    /** Regularization cost for the word vectors */
    private double regWordVector = 0.0001f;

    private int inputMiniBatchSize;

    /**
     * How many epochs between resets of the adagrad learning rates.
     * Set to 0 to never reset.
     */
    private int adagradResetFrequency = 1;

    /** Regularization cost for the applyTransformToOrigin INDArray  */
    private double regTransformINDArray = 0.001f;

    /**
     * Nx2N+1, where N is the size of the word vectors
     */
    private MultiDimensionalMap<String, String, INDArray> binaryTransform;

    /**
     * 2Nx2NxN, where N is the size of the word vectors
     */
    private MultiDimensionalMap<String, String, INDArray> binaryTensors;

    /**
     * CxN+1, where N = size of word vectors, C is the number of classes
     */
    private Map<String, INDArray> unaryClassification;

    private WeightLookupTable<VocabWord> featureVectors;
    private VocabCache<VocabWord> vocabCache;

    /**
     * CxN+1, where N = size of word vectors, C is the number of classes
     */
    private MultiDimensionalMap<String, String, INDArray> binaryClassification;


    /**
     * Cached here for easy calculation of the model size;
     * MultiDimensionalMap does not return that in O(1) time
     */
    private  int numBinaryMatrices;

    /** How many elements a transformation matrix has */
    private  int binaryTransformSize;
    /** How many elements the binary transformation INd4j have */
    private  int binaryINd4jize;
    /** How many elements a classification matrix has */
    private  int binaryClassificationSize;

    /**
     * Cached here for easy calculation of the model size;
     * MultiDimensionalMap does not return that in O(1) time
     */
    private  int numUnaryMatrices;

    /** How many elements a classification matrix has */
    private  int unaryClassificationSize;

    private INDArray identity;

    private Map<Integer,Double> classWeights;

    private static final Logger log = LoggerFactory.getLogger(RNTN.class);


    private transient ActorSystem rnTnActorSystem = ActorSystem.create("RNTN");
    protected int index=0;



    private RNTN(int numHidden,
                 Random rng,
                 boolean useDoubleTensors,
                 boolean combineClassification,
                 boolean simplifiedModel,
                 boolean randomFeatureVectors,
                 double scalingForInit,
                 boolean lowerCasefeatureNames,
                 String activationFunction,
                 int adagradResetFrequency,
                 double regTransformINDArray,
                 WeightLookupTable featureVectors,
                 VocabCache vocabCache,
                 int numBinaryMatrices,
                 int binaryTransformSize,
                 int binaryINd4jize,
                 int binaryClassificationSize,
                 int numUnaryMatrices,
                 int unaryClassificationSize,
                 Map<Integer, Double> classWeights) {
        this.vocabCache = vocabCache;
        this.numHidden = numHidden;
        this.rng = rng;
        this.useDoubleTensors = useDoubleTensors;
        this.combineClassification = combineClassification;
        this.simplifiedModel = simplifiedModel;
        this.randomFeatureVectors = randomFeatureVectors;
        this.scalingForInit = scalingForInit;
        this.lowerCasefeatureNames = lowerCasefeatureNames;
        this.activationFunction = activationFunction;
        this.adagradResetFrequency = adagradResetFrequency;
        this.regTransformINDArray = regTransformINDArray;
        this.featureVectors = featureVectors;
        this.numBinaryMatrices = numBinaryMatrices;
        this.binaryTransformSize = binaryTransformSize;
        this.binaryINd4jize = binaryINd4jize;
        this.binaryClassificationSize = binaryClassificationSize;
        this.numUnaryMatrices = numUnaryMatrices;
        this.unaryClassificationSize = unaryClassificationSize;
        this.classWeights = classWeights;
        init();
    }


    private void init() {

        if(rng == null) {
            rng = Nd4j.getRandom();
        }
        MultiDimensionalSet<String, String> binaryProductions = MultiDimensionalSet.hashSet();
        if (simplifiedModel) {
            binaryProductions.add("", "");
        } else {
            // TODO
            // figure out what binary productions we have in these trees
            // Note: the current sentiment training data does not actually
            // have any constituent labels
            throw new UnsupportedOperationException("Not yet implemented");
        }

        Set<String> unaryProductions = new HashSet<>();

        if (simplifiedModel) {
            unaryProductions.add("");
        } else {
            // TODO
            // figure out what unary productions we have in these trees (preterminals only, after the collapsing)
            throw new UnsupportedOperationException("Not yet implemented");
        }


        identity = Nd4j.eye(numHidden);

        binaryTransform = MultiDimensionalMap.newTreeBackedMap();
        binaryTensors = MultiDimensionalMap.newTreeBackedMap();
        binaryClassification = MultiDimensionalMap.newTreeBackedMap();

        // When making a flat model (no semantic untying) the
        // basicCategory function will return the same basic category for
        // all labels, so all entries will map to the same matrix
        for (Pair<String, String> binary : binaryProductions) {
            String left = basicCategory(binary.getFirst());
            String right = basicCategory(binary.getSecond());
            if (binaryTransform.contains(left, right)) {
                continue;
            }

            binaryTransform.put(left, right, randomTransformMatrix());
            if (useDoubleTensors) {
                binaryTensors.put(left, right, randomBinaryINDArray());
            }

            if (!combineClassification) {
                binaryClassification.put(left, right, randomClassificationMatrix());
            }
        }

        numBinaryMatrices = binaryTransform.size();
        binaryTransformSize = numHidden * (2 * numHidden + 1);

        if (useDoubleTensors) {
            binaryINd4jize = numHidden * numHidden * numHidden * 4;
        } else {
            binaryINd4jize = 0;
        }

        binaryClassificationSize = (combineClassification) ? 0 : numOuts * (numHidden + 1);

        unaryClassification = new TreeMap<>();

        // When making a flat model (no semantic untying) the
        // basicCategory function will return the same basic category for
        // all labels, so all entries will map to the same matrix

        for (String unary : unaryProductions) {
            unary = basicCategory(unary);
            if (unaryClassification.containsKey(unary)) {
                continue;
            }
            unaryClassification.put(unary, randomClassificationMatrix());
        }


        binaryClassificationSize = (combineClassification) ? 0 : numOuts * (numHidden + 1);

        numUnaryMatrices = unaryClassification.size();
        unaryClassificationSize = numOuts * (numHidden + 1);



        numUnaryMatrices = unaryClassification.size();
        unaryClassificationSize = numOuts * (numHidden + 1);
        classWeights = new HashMap<>();

    }

    @Override
    public int getIndex() {
        return index;
    }

    @Override
    public void setInput(INDArray input) {

    }

    @Override
    public void setIndex(int index) {
        this.index = index;
    }

    public Collection<IterationListener> getListeners() {
        return iterationListeners;
    }

    @Override
    public void setListeners(IterationListener... listeners) {

    }

    public void setListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
    }

    INDArray randomBinaryINDArray() {
        double range = 1.0f / (4.0f * numHidden);
        INDArray ret = Nd4j.rand(new int[]{numHidden,numHidden * 2, numHidden * 2}, -range, range, rng);
        return ret.muli(scalingForInit);
    }

    public INDArray randomTransformMatrix() {
        INDArray binary = Nd4j.create(numHidden, numHidden * 2 + 1);
        // bias column values are initialized zero
        INDArray block = randomTransformBlock();
        INDArrayIndex[] indices = new INDArrayIndex[] {interval(0,block.rows()),interval(0,block.columns())};
        binary.put(indices,block);
        INDArrayIndex[] indices2 = new INDArrayIndex[]{interval(0,block.rows()),interval(numHidden,numHidden + block.columns())};
        binary.put(indices2,randomTransformBlock());
        Nd4j.getBlasWrapper().level1().scal(binary.length(),scalingForInit,binary);
        return binary;
    }

    public INDArray randomTransformBlock() {
        double range = 1.0 /  (Math.sqrt((double) numHidden) * 2.0f);
        INDArray ret = Nd4j.rand(numHidden,numHidden,-range,range,rng).addi(identity);
        return ret;
    }

    /**
     * Returns matrices of the right size for either binary or unary (terminal) classification
     */
    INDArray randomClassificationMatrix() {
        // Leave the bias column with 0 values
        double range = 1.0 / (Math.sqrt((double) numHidden));
        INDArray ret = Nd4j.zeros(numOuts,numHidden + 1);
        INDArray insert = Nd4j.rand(numOuts, numHidden, -range, range, rng);
        ret.put(new INDArrayIndex[] {interval(0,numOuts),interval(0,numHidden)},insert);
        Nd4j.getBlasWrapper().level1().scal(ret.length(), scalingForInit, ret);
        return ret;
    }

    /**
     *
     * Shut down this network actor
     */
    public void shutdown() {
        rnTnActorSystem.shutdown();
    }

    /**
     * Trains the network on this mini batch and waits for the training set to complete
     * @param trainingBatch the trees to iterate on
     */
    public void fit(List<Tree> trainingBatch) {
        final CountDownLatch c = new CountDownLatch(trainingBatch.size());

        List<Future<Object>> futureBatch = fitAsync(trainingBatch);

        for(Future<Object> f : futureBatch) {
            f.onComplete(new OnComplete<Object>() {
                @Override
                public void onComplete(Throwable throwable, Object e) throws Throwable {
                    if (throwable != null)
                        log.warn("Error occurred training batch", throwable);

                    c.countDown();
                }
            }, rnTnActorSystem.dispatcher());
        }


        try {
            c.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Trains the network on this mini batch and returns a list of futures for each training job
     * @param trainingBatch the trees to iterate on
     */
    public List<Future<Object>> fitAsync(final List<Tree> trainingBatch) {
        int count = 0;

        List<Future<Object>> futureBatch = new ArrayList<>();

        for(final Tree t : trainingBatch) {
            log.info("Working mini batch " + count++);
            futureBatch.add(Futures.future(new Callable<Object>() {
                @Override
                public Object call() throws Exception {
                    forwardPropagateTree(t);
                    try {
                        INDArray params = getParameters();
                        INDArray gradient = getValueGradient(trainingBatch);
                        if(params.length() != gradient.length())
                            throw new IllegalStateException("Params not equal to gradient!");
                        setParams(params.subi(gradient));
                    }catch(NegativeArraySizeException e) {
                        log.warn("Couldnt compute parameters due to negative array size...for trees " + t);
                    }

                    return null;
                }
            },rnTnActorSystem.dispatcher()));


        }
        return futureBatch;
    }


    public INDArray getWForNode(Tree node) {
        if (node.children().size() == 2) {
            String leftLabel = node.children().get(0).value();
            String leftBasic = basicCategory(leftLabel);
            String rightLabel = node.children().get(1).value();
            String rightBasic = basicCategory(rightLabel);
            return binaryTransform.get(leftBasic, rightBasic);
        } else if (node.children().size() == 1) {
            throw new AssertionError("No unary applyTransformToOrigin matrices, only unary classification");
        } else {
            throw new AssertionError("Unexpected tree children size of " + node.children().size());
        }
    }

    public INDArray getINDArrayForNode(Tree node) {
        if (!useDoubleTensors) {
            throw new AssertionError("Not using tensors");
        }
        if (node.children().size() == 2) {
            String leftLabel = node.children().get(0).value();
            String leftBasic = basicCategory(leftLabel);
            String rightLabel = node.children().get(1).value();
            String rightBasic = basicCategory(rightLabel);
            return binaryTensors.get(leftBasic, rightBasic);
        } else if (node.children().size() == 1) {
            throw new AssertionError("No unary transform matrices, only unary classification");
        } else {
            throw new AssertionError("Unexpected tree children size of " + node.children().size());
        }
    }

    public INDArray getClassWForNode(Tree node) {
        if (combineClassification) {
            return unaryClassification.get("");
        } else if (node.children().size() == 2) {
            String leftLabel = node.children().get(0).value();
            String leftBasic = basicCategory(leftLabel);
            String rightLabel = node.children().get(1).value();
            String rightBasic = basicCategory(rightLabel);
            return binaryClassification.get(leftBasic, rightBasic);
        } else if (node.children().size() == 1) {
            String unaryLabel = node.children().get(0).value();
            String unaryBasic = basicCategory(unaryLabel);
            return unaryClassification.get(unaryBasic);
        } else {
            throw new AssertionError("Unexpected tree children size of " + node.children().size());
        }
    }



    private INDArray getINDArrayGradient(INDArray deltaFull, INDArray leftVector, INDArray rightVector) {
        int size = deltaFull.length();
        INDArray Wt_df = Nd4j.create(size,size * 2, size*2);
        INDArray fullVector = Nd4j.concat(0,leftVector, rightVector);
        for (int slice = 0; slice < size; slice++) {
            Nd4j.getBlasWrapper().level1().scal(deltaFull.length(), deltaFull.getScalar(slice).getFloat(0), fullVector);
            Wt_df.putSlice(slice, fullVector.mmul(fullVector.transpose()));

        }
        return Wt_df;
    }



    public INDArray getFeatureVector(String word) {
        INDArray ret = featureVectors.vector(getVocabWord(word));
        if(ret.isRowVector())
            ret = ret.transpose();
        return ret;
    }

    public String getVocabWord(String word) {
        if (lowerCasefeatureNames) {
            word = word.toLowerCase();
        }
        if (vocabCache.containsWord(word)) {
            return word;
        }
        // TODO: go through unknown words here
        return Word2Vec.DEFAULT_UNK;
    }

    public String basicCategory(String category) {
        if (simplifiedModel) {
            return "";
        }
        throw new IllegalStateException("Only simplified model enabled");
    }

    public INDArray getUnaryClassification(String category) {
        category = basicCategory(category);
        return unaryClassification.get(category);
    }

    public INDArray getBinaryClassification(String left, String right) {
        if (combineClassification) {
            return unaryClassification.get("");
        } else {
            left = basicCategory(left);
            right = basicCategory(right);
            return binaryClassification.get(left, right);
        }
    }

    public INDArray getBinaryTransform(String left, String right) {
        left = basicCategory(left);
        right = basicCategory(right);
        return binaryTransform.get(left, right);
    }

    public INDArray getBinaryINDArray(String left, String right) {
        left = basicCategory(left);
        right = basicCategory(right);
        return binaryTensors.get(left, right);
    }

    double scaleAndRegularize(MultiDimensionalMap<String, String, INDArray> derivatives,
                              MultiDimensionalMap<String, String, INDArray> currentMatrices,
                              double scale,
                              double regCost) {

        double cost = 0.0f; // the regularization cost
        for (MultiDimensionalMap.Entry<String, String, INDArray> entry : currentMatrices.entrySet()) {
            INDArray D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
            if(D.data().dataType() == DataBuffer.Type.DOUBLE)
                D = Nd4j.getBlasWrapper().scal(scale,D).addi(Nd4j.getBlasWrapper().scal(regCost, entry.getValue()));
            else
                D = Nd4j.getBlasWrapper().scal((float) scale,D).addi(Nd4j.getBlasWrapper().scal((float) regCost, entry.getValue()));

            derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
            cost +=  entry.getValue().mul(entry.getValue()).sum(Integer.MAX_VALUE).getDouble(0) * regCost / 2.0;
        }
        return cost;
    }

    double scaleAndRegularize(Map<String, INDArray> derivatives,
                              Map<String,INDArray> currentMatrices,
                              double scale,
                              double regCost) {

        double cost = 0.0f; // the regularization cost
        for (String s  : currentMatrices.keySet()) {
            INDArray D = derivatives.get(s);
            INDArray vector = currentMatrices.get(s);
            if(D.data().dataType() == DataBuffer.Type.DOUBLE)
                D = Nd4j.getBlasWrapper().scal(scale,D).addi(Nd4j.getBlasWrapper().scal(regCost,vector));
            else
                D = Nd4j.getBlasWrapper().scal((float) scale,D).addi(Nd4j.getBlasWrapper().scal((float) regCost,vector));

            derivatives.put(s, D);
            cost += vector.mul(vector).sum(Integer.MAX_VALUE).getDouble(0) * regCost / 2.0f;
        }
        return cost;
    }


    double scaleAndRegularize(Map<String, INDArray> derivatives,
                              WeightLookupTable currentMatrices,
                              double scale,
                              double regCost) {

        double cost = 0.0f; // the regularization cost
        for (String s  : vocabCache.words()) {
            INDArray D = derivatives.get(s);
            INDArray vector = currentMatrices.vector(s);
            if(D.data().dataType() == DataBuffer.Type.DOUBLE)
                D = Nd4j.getBlasWrapper().scal(scale,D).addi(Nd4j.getBlasWrapper().scal(regCost,vector));
            else
                D = Nd4j.getBlasWrapper().scal((float) scale,D).addi(Nd4j.getBlasWrapper().scal((float) regCost,vector));

            derivatives.put(s, D);
            cost += vector.mul(vector).sum(Integer.MAX_VALUE).getDouble(0) * regCost / 2.0f;
        }
        return cost;
    }

    double scaleAndRegularizeINDArray(MultiDimensionalMap<String, String, INDArray> derivatives,
                                      MultiDimensionalMap<String, String, INDArray> currentMatrices,
                                      double scale,
                                      double regCost) {
        double cost = 0.0f; // the regularization cost
        for (MultiDimensionalMap.Entry<String, String, INDArray> entry : currentMatrices.entrySet()) {
            INDArray D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
            D = D.muli(scale).add(entry.getValue().muli(regCost));
            derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
            cost += entry.getValue().mul(entry.getValue()).sum(Integer.MAX_VALUE).getDouble(0) * regCost / 2.0f;
        }
        return cost;
    }

    private void backpropDerivativesAndError(Tree tree,
                                             MultiDimensionalMap<String, String, INDArray> binaryTD,
                                             MultiDimensionalMap<String, String, INDArray> binaryCD,
                                             MultiDimensionalMap<String, String, INDArray> binaryINDArrayTD,
                                             Map<String, INDArray> unaryCD,
                                             Map<String, INDArray> wordVectorD) {

        INDArray delta = Nd4j.create(numHidden, 1);
        backpropDerivativesAndError(tree, binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD, delta);
    }

    private void backpropDerivativesAndError(Tree tree,
                                             MultiDimensionalMap<String, String, INDArray> binaryTD,
                                             MultiDimensionalMap<String, String, INDArray> binaryCD,
                                             MultiDimensionalMap<String, String, INDArray> binaryINDArrayTD,
                                             Map<String, INDArray> unaryCD,
                                             Map<String, INDArray> wordVectorD,
                                             INDArray deltaUp) {
        if (tree.isLeaf()) {
            return;
        }

        INDArray currentVector = tree.vector();
        String category = tree.label();
        category = basicCategory(category);

        // Build a vector that looks like 0,0,1,0,0 with an indicator for the correct class
        INDArray goldLabel = Nd4j.create(numOuts, 1);
        int goldClass = tree.goldLabel();
        if (goldClass >= 0) {
            assert goldClass <= numOuts : "Tried adding a label that was >= to the number of configured outputs " + numOuts + " with label " + goldClass;
            goldLabel.putScalar(goldClass, 1.0f);
        }

        Double nodeWeight = classWeights.get(goldClass);
        if(nodeWeight == null)
            nodeWeight = 1.0;
        INDArray predictions = tree.prediction();

        // If this is an unlabeled class, transform deltaClass to 0.  We could
        // make this more efficient by eliminating various of the below
        // calculations, but this would be the easiest way to handle the
        // unlabeled class
        INDArray deltaClass = null;
        if(predictions.data().dataType() == DataBuffer.Type.DOUBLE) {
            deltaClass =  goldClass >= 0 ? Nd4j.getBlasWrapper().scal(nodeWeight,predictions.sub(goldLabel)) : Nd4j.create(predictions.rows(), predictions.columns());

        }
        else {
            deltaClass =  goldClass >= 0 ? Nd4j.getBlasWrapper().scal((float) nodeWeight.doubleValue(),predictions.sub(goldLabel)) : Nd4j.create(predictions.rows(), predictions.columns());

        }
        INDArray localCD = deltaClass.mmul(Nd4j.appendBias(currentVector).transpose());

        double error = - (Transforms.log(predictions).muli(goldLabel).sum(Integer.MAX_VALUE).getDouble(0));
        error = error * nodeWeight;
        tree.setError(error);

        if (tree.isPreTerminal()) { // below us is a word vector
            unaryCD.put(category, unaryCD.get(category).add(localCD));

            String word = tree.children().get(0).label();
            word = getVocabWord(word);


            INDArray currentVectorDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction,currentVector));
            INDArray deltaFromClass = getUnaryClassification(category).transpose().mmul(deltaClass);
            deltaFromClass = deltaFromClass.get(interval(0, numHidden),interval(0, 1)).mul(currentVectorDerivative);
            INDArray deltaFull = deltaFromClass.add(deltaUp);
            INDArray wordVector = wordVectorD.get(word);
            wordVectorD.put(word, wordVector.add(deltaFull));


        } else {
            // Otherwise, this must be a binary node
            String leftCategory = basicCategory(tree.children().get(0).label());
            String rightCategory = basicCategory(tree.children().get(1).label());
            if (combineClassification) {
                unaryCD.put("", unaryCD.get("").add(localCD));
            } else {
                binaryCD.put(leftCategory, rightCategory, binaryCD.get(leftCategory, rightCategory).add(localCD));
            }

            INDArray currentVectorDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, currentVector));
            INDArray deltaFromClass = getBinaryClassification(leftCategory, rightCategory).transpose().mmul(deltaClass);

            INDArray mult = deltaFromClass.get(interval(0, numHidden),interval(0, 1));
            deltaFromClass = mult.muli(currentVectorDerivative);
            INDArray deltaFull = deltaFromClass.add(deltaUp);

            INDArray leftVector =tree.children().get(0).vector();
            INDArray rightVector = tree.children().get(1).vector();

            INDArray childrenVector = Nd4j.appendBias(leftVector,rightVector);

            //deltaFull 50 x 1, childrenVector: 50 x 2
            INDArray add = binaryTD.get(leftCategory, rightCategory);

            INDArray W_df = deltaFromClass.mmul(childrenVector.transpose());
            binaryTD.put(leftCategory, rightCategory, add.add(W_df));

            INDArray deltaDown;
            if (useDoubleTensors) {
                INDArray Wt_df = getINDArrayGradient(deltaFull, leftVector, rightVector);
                binaryINDArrayTD.put(leftCategory, rightCategory, binaryINDArrayTD.get(leftCategory, rightCategory).add(Wt_df));
                deltaDown = computeINDArrayDeltaDown(deltaFull, leftVector, rightVector, getBinaryTransform(leftCategory, rightCategory), getBinaryINDArray(leftCategory, rightCategory));
            } else {
                deltaDown = getBinaryTransform(leftCategory, rightCategory).transpose().mmul(deltaFull);
            }

            INDArray leftDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, leftVector));
            INDArray rightDerivative =Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, rightVector));
            INDArray leftDeltaDown = deltaDown.get(interval(0, deltaFull.rows()),interval( 0, 1));
            INDArray rightDeltaDown = deltaDown.get(interval(deltaFull.rows(), deltaFull.rows() * 2),interval( 0, 1));
            backpropDerivativesAndError(tree.children().get(0), binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD, leftDerivative.mul(leftDeltaDown));
            backpropDerivativesAndError(tree.children().get(1), binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD, rightDerivative.mul(rightDeltaDown));
        }
    }

    private INDArray computeINDArrayDeltaDown(INDArray deltaFull, INDArray leftVector, INDArray rightVector,
                                              INDArray W, INDArray Wt) {
        INDArray WTDelta = W.transpose().mmul(deltaFull);
        INDArray WTDeltaNoBias = WTDelta.isMatrix() ? WTDelta.get(interval( 0, 1),interval(0, (deltaFull.rows() * 2) + 1)) :
                WTDelta.get(interval(0, (deltaFull.rows() * 2)));
        int size = deltaFull.length();
        INDArray deltaINDArray = Nd4j.create(size * 2, 1);
        INDArray fullVector = Nd4j.concat(0, leftVector, rightVector);
        for (int slice = 0; slice < size; ++slice) {
            if(deltaFull.data().dataType() == DataBuffer.Type.DOUBLE) {
                INDArray scaledFullVector = Nd4j.getBlasWrapper().scal(deltaFull.getScalar(slice).getDouble(0),fullVector);
                deltaINDArray = deltaINDArray.add(Wt.slice(slice).add(Wt.slice(slice).transpose()).mmul(scaledFullVector));
            }
            else {
                INDArray scaledFullVector = Nd4j.getBlasWrapper().scal((float) deltaFull.getScalar(slice).getDouble(0),fullVector);
                deltaINDArray.addi(Wt.slice(slice).add(Wt.slice(slice).transpose()).mmul(scaledFullVector));
            }

        }
        return deltaINDArray.add(WTDeltaNoBias);
    }


    /**
     * This is the method to call for assigning labels and node vectors
     * to the Tree.  After calling this, each of the non-leaf nodes will
     * have the node vector and the predictions of their classes
     * assigned to that subtree's node.
     */
    public void forwardPropagateTree(Tree tree) {
        INDArray nodeVector;
        INDArray classification;

        if (tree.isLeaf()) {
            // We do nothing for the leaves.  The preterminals will
            // calculate the classification for this word/tag.  In fact, the
            // recursion should not have gotten here (unless there are
            // degenerate trees of just one leaf)
            throw new AssertionError("We should not have reached leaves in forwardPropagate");
        }

        else if (tree.isPreTerminal()) {
            classification = getUnaryClassification(tree.label());
            String word = tree.children().get(0).value();
            INDArray wordVector = getFeatureVector(word);
            if(wordVector == null) {
                wordVector = featureVectors.vector(Word2Vec.DEFAULT_UNK);
            }


            nodeVector = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, wordVector));
        } else if (tree.children().size() == 1) {
            throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
        } else if (tree.children().size() == 2) {
            Tree left = tree.firstChild(),right = tree.lastChild();
            forwardPropagateTree(left);
            forwardPropagateTree(right);

            String leftCategory = tree.children().get(0).label();
            String rightCategory = tree.children().get(1).label();
            INDArray W = getBinaryTransform(leftCategory, rightCategory);
            classification = getBinaryClassification(leftCategory, rightCategory);

            INDArray leftVector = tree.children().get(0).vector();
            INDArray rightVector = tree.children().get(1).vector();

            INDArray childrenVector = Nd4j.appendBias(leftVector,rightVector);


            if (useDoubleTensors) {
                INDArray doubleT = getBinaryINDArray(leftCategory, rightCategory);
                INDArray INDArrayIn = Nd4j.concat(0,leftVector, rightVector);
                INDArray INDArrayOut = Nd4j.bilinearProducts(doubleT,INDArrayIn);
                nodeVector = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, W.mmul(childrenVector).addi(INDArrayOut)));

            }

            else
                nodeVector = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunction, W.mmul(childrenVector)));

        } else {
            throw new AssertionError("Tree not correctly binarized");
        }

        INDArray inputWithBias  = Nd4j.appendBias(nodeVector);
        if(inputWithBias.rows() != classification.columns())
            inputWithBias = inputWithBias.transpose();
        INDArray preAct = classification.mmul(inputWithBias);
        INDArray predictions = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(outputActivation, preAct));

        tree.setPrediction(predictions);
        tree.setVector(nodeVector);
    }



    private INDArray getDoubleTensorGradient(INDArray deltaFull, INDArray leftVector, INDArray rightVector) {
        int size = deltaFull.length();
        INDArray Wt_df = Nd4j.create(size * 2, size * 2, size);
        INDArray fullVector = Nd4j.concat(0,leftVector, rightVector);
        for (int slice = 0; slice < size; ++slice) {
            Nd4j.getBlasWrapper().level1().scal(fullVector.length(),deltaFull.getDouble(slice),fullVector);
            Wt_df.putSlice(slice, fullVector.mmul(fullVector.transpose()));

        }
        return Wt_df;
    }


    /**
     * output the prediction probabilities for each tree
     * @param trees the trees to predict
     * @return the prediction probabilities for each tree
     */
    public List<INDArray> output(List<Tree> trees) {
        List<INDArray> ret = new ArrayList<>();
        for(Tree t : trees) {
            forwardPropagateTree(t);
            ret.add(t.prediction());
        }

        return ret;
    }


    /**
     * output the top level labels for each tree
     * @param trees the trees to predict
     * @return the prediction labels for each tree
     */
    public List<Integer> predict(List<Tree> trees) {
        List<Integer> ret = new ArrayList<>();
        for(Tree t : trees) {
            forwardPropagateTree(t);
            ret.add(Nd4j.getBlasWrapper().iamax(t.prediction()));
        }

        return ret;
    }

    /**
     * Set the parameters for the model
     * @param params
     */
    public void setParams(INDArray params) {
        if(params.length() != getNumParameters())
            throw new IllegalStateException("Unable to set parameters of length " + params.length() + " must be of length " + numParameters);
        Nd4j.setParams(params,
                binaryTransform.values().iterator(),
                binaryClassification.values().iterator(),
                binaryTensors.values().iterator(),
                unaryClassification.values().iterator(),
                featureVectors.vectors());
        computeGradientAndScore();
    }

    @Override
    public void applyLearningRateScoreDecay() {

    }

    public int getNumParameters() {
        if(numParameters < 0) {
            int totalSize = 0;
            List<Iterator<INDArray>> list = Arrays.asList(binaryTransform.values().iterator(),
                    binaryClassification.values().iterator(),
                    binaryTensors.values().iterator(),
                    unaryClassification.values().iterator(),
                    featureVectors.vectors());
            for(Iterator<INDArray> iter : list) {
                while(iter.hasNext())
                    totalSize += iter.next().length();
            }

            numParameters = totalSize;
        }

        return numParameters;
    }


    public INDArray getParameters() {
        return Nd4j.toFlattened(
                getNumParameters(),
                binaryTransform.values().iterator(),
                binaryClassification.values().iterator(),
                binaryTensors.values().iterator(),
                unaryClassification.values().iterator(),
                featureVectors.vectors());
    }


    public INDArray getValueGradient(final List<Tree> trainingBatch) {


        // We use TreeMap for each of these so that they stay in a
        // canonical sorted order
        // TODO: factor out the initialization routines
        // binaryTD stands for Transform Derivatives
        final MultiDimensionalMap<String, String, INDArray> binaryTD = MultiDimensionalMap.newTreeBackedMap();
        // the derivatives of the INd4j for the binary nodes
        final MultiDimensionalMap<String, String, INDArray> binaryINDArrayTD = MultiDimensionalMap.newTreeBackedMap();
        // binaryCD stands for Classification Derivatives
        final MultiDimensionalMap<String, String, INDArray> binaryCD = MultiDimensionalMap.newTreeBackedMap();

        // unaryCD stands for Classification Derivatives
        final Map<String, INDArray> unaryCD = new TreeMap<>();

        // word vector derivatives
        final Map<String, INDArray> wordVectorD = new TreeMap<>();

        for (MultiDimensionalMap.Entry<String, String, INDArray> entry : binaryTransform.entrySet()) {
            int numRows = entry.getValue().rows();
            int numCols = entry.getValue().columns();

            binaryTD.put(entry.getFirstKey(), entry.getSecondKey(), Nd4j.create(numRows, numCols));
        }

        if (!combineClassification) {
            for (MultiDimensionalMap.Entry<String, String, INDArray> entry :  binaryClassification.entrySet()) {
                int numRows = entry.getValue().rows();
                int numCols = entry.getValue().columns();

                binaryCD.put(entry.getFirstKey(), entry.getSecondKey(), Nd4j.create(numRows, numCols));
            }
        }

        if (useDoubleTensors) {
            for (MultiDimensionalMap.Entry<String, String, INDArray> entry : binaryTensors.entrySet()) {
                int numRows = entry.getValue().size(1);
                int numCols = entry.getValue().size(2);
                int numSlices = entry.getValue().slices();

                binaryINDArrayTD.put(entry.getFirstKey(), entry.getSecondKey(), Nd4j.create(numRows, numCols, numSlices));
            }
        }

        for (Map.Entry<String, INDArray> entry : unaryClassification.entrySet()) {
            int numRows = entry.getValue().rows();
            int numCols = entry.getValue().columns();
            unaryCD.put(entry.getKey(), Nd4j.create(numRows, numCols));
        }


        for (String s : vocabCache.words()) {
            INDArray vector = featureVectors.vector(s);
            int numRows = vector.rows();
            int numCols = vector.columns();
            wordVectorD.put(s, Nd4j.create(numRows, numCols));
        }


        final List<Tree> forwardPropTrees = new CopyOnWriteArrayList<>();
        //if(!forwardPropTrees.isEmpty())
        Parallelization.iterateInParallel(trainingBatch,new Parallelization.RunnableWithParams<Tree>() {

            public void run(Tree currentItem, Object[] args) {
                Tree trainingTree = new Tree(currentItem);
                trainingTree.connect(new ArrayList<>(currentItem.children()));
                // this will attach the error vectors and the node vectors
                // to each node in the tree
                forwardPropagateTree(trainingTree);
                forwardPropTrees.add(trainingTree);

            }
        },rnTnActorSystem);


        // TODO: we may find a big speedup by separating the derivatives and then summing
        final AtomicDouble error = new AtomicDouble(0);
        if(!forwardPropTrees.isEmpty())
            Parallelization.iterateInParallel(forwardPropTrees,new Parallelization.RunnableWithParams<Tree>() {

                public void run(Tree currentItem, Object[] args) {
                    backpropDerivativesAndError(currentItem, binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD);
                    error.addAndGet(currentItem.errorSum());

                }
            },new Parallelization.RunnableWithParams<Tree>() {

                public void run(Tree currentItem, Object[] args) {
                }
            },rnTnActorSystem,new Object[]{ binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD});



        // scale the error by the number of sentences so that the
        // regularization isn't drowned out for large training batchs
        double scale = trainingBatch == null || trainingBatch.isEmpty() ? 1.0f :  (1.0f / trainingBatch.size());
        value = error.doubleValue() * scale;

        value += scaleAndRegularize(binaryTD, binaryTransform, scale, regTransformMatrix);
        value += scaleAndRegularize(binaryCD, binaryClassification, scale, regClassification);
        value += scaleAndRegularizeINDArray(binaryINDArrayTD, binaryTensors, scale, regTransformINDArray);
        value += scaleAndRegularize(unaryCD, unaryClassification, scale, regClassification);
        value += scaleAndRegularize(wordVectorD, featureVectors, scale, regWordVector);

        INDArray derivative = Nd4j.toFlattened(
                getNumParameters(),
                binaryTD.values().iterator(),
                binaryCD.values().iterator(),
                binaryINDArrayTD.values().iterator()
                , unaryCD.values().iterator(),
                wordVectorD.values().iterator());

        if(derivative.length() != numParameters)
            throw new IllegalStateException("Gradient has wrong number of parameters " + derivative.length() + " should have been " + numParameters);

        if(paramAdaGrad == null)
            paramAdaGrad = new AdaGrad(1,derivative.columns());

        derivative = paramAdaGrad.getGradient(derivative,0);

        return derivative;
    }


    public double getValue() {
        return value;
    }

    @Override
    public void fit() {
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double score() {
        return 0;
    }


    @Override
    public INDArray params() {
        return getParameters();
    }

    @Override
    public int numParams() {
        return getNumParameters();
    }

    @Override
    public int numParams(boolean backwards) {
        return 0;
    }


    @Override
    public void fit(INDArray data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException();

    }

    @Override
    public Gradient gradient() {
        return null;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return null;
    }

    @Override
    public int batchSize() {
        return 0;
    }

    @Override
    public double calcL2() {
        return 0;
    }

    @Override
    public double calcL1() {
        return 0;
    }

    @Override
    public Layer.Type type() {
        return Layer.Type.RECURSIVE;
    }

    @Override
    public Gradient error(INDArray input) {
        return null;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        return null;
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        return null;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return null;
    }

    @Override
    public void merge(Layer layer, int batchSize) {

    }

    @Override
    public INDArray getParam(String param) {
        return null;
    }

    @Override
    public void initParams() {

    }

    @Override
    public Map<String, INDArray> paramTable() {
        return null;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {

    }

    @Override
    public void setParam(String key, INDArray val) {

    }

    @Override
    public void clear() {

    }

    @Override
    public INDArray activationMean() {
        return null;
    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public NeuralNetConfiguration conf() {
        return null;
    }


    @Override
    public INDArray preOutput(INDArray x) {
        return null;
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return null;
    }

    @Override
    public INDArray activate(TrainingMode training) {
        return null;
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        return null;
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return null;
    }

    @Override
    public INDArray activate(boolean training) {
        return null;
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        return null;
    }

    @Override
    public INDArray activate() {
        return null;
    }

    @Override
    public INDArray activate(INDArray input) {
        return null;
    }

    @Override
    public Layer transpose() {
        return null;
    }

    @Override
    public Layer clone() {
        return null;
    }

    @Override
    public void computeGradientAndScore() {

    }

    @Override
    public void accumulateScore(double accum) {

    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {

    }

    @Override
    public INDArray input() {
        return null;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return null;
    }

    public static class Builder {
        //must be same size as word vectors
        private int numHidden;
        private Random rng;
        private boolean useINd4j;
        private boolean combineClassification = true;
        private boolean simplifiedModel = true;
        private boolean randomFeatureVectors;
        private double scalingForInit = 1e-3f;
        private boolean lowerCasefeatureNames;
        private String activationFunction = "sigmoid",
                outputActivationFunction = "softmax";
        private int adagradResetFrequency;
        private double regTransformINDArray;
        private WeightLookupTable featureVectors;
        private int numBinaryMatrices;
        private int binaryTransformSize;
        private int binaryINd4jize;
        private int binaryClassificationSize;
        private int numUnaryMatrices;
        private int unaryClassificationSize;
        private Map<Integer, Double> classWeights;
        private VocabCache<VocabWord> vocabCache;



        public Builder vocabCache(VocabCache vocabCache) {
            this.vocabCache = vocabCache;
            return this;
        }



        public Builder withOutputActivation(String outputActivationFunction) {
            this.outputActivationFunction = outputActivationFunction;
            return this;
        }

        public Builder setFeatureVectors(Word2Vec vec) {
            vocabCache = vec.vocab();
            return setFeatureVectors(vec.lookupTable());
        }

        public Builder setNumHidden(int numHidden) {
            this.numHidden = numHidden;
            return this;
        }

        public Builder setRng(Random rng) {
            this.rng = rng;
            return this;
        }

        public Builder setUseTensors(boolean useINd4j) {
            this.useINd4j = useINd4j;
            return this;
        }

        public Builder setCombineClassification(boolean combineClassification) {
            this.combineClassification = combineClassification;
            return this;
        }

        public Builder setSimplifiedModel(boolean simplifiedModel) {
            this.simplifiedModel = simplifiedModel;
            return this;
        }

        public Builder setRandomFeatureVectors(boolean randomFeatureVectors) {
            this.randomFeatureVectors = randomFeatureVectors;
            return this;
        }

        public Builder setScalingForInit(double scalingForInit) {
            this.scalingForInit = scalingForInit;
            return this;
        }

        public Builder setLowerCasefeatureNames(boolean lowerCasefeatureNames) {
            this.lowerCasefeatureNames = lowerCasefeatureNames;
            return this;
        }

        public Builder setActivationFunction(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }



        public Builder setAdagradResetFrequency(int adagradResetFrequency) {
            this.adagradResetFrequency = adagradResetFrequency;
            return this;
        }

        public Builder setRegTransformINDArray(double regTransformINDArray) {
            this.regTransformINDArray = regTransformINDArray;
            return this;
        }

        public Builder setFeatureVectors(WeightLookupTable<VocabWord> featureVectors) {
            this.featureVectors = featureVectors;
            this.numHidden = featureVectors.vectors().next().columns();
            return this;
        }

        public Builder setNumBinaryMatrices(int numBinaryMatrices) {
            this.numBinaryMatrices = numBinaryMatrices;
            return this;
        }

        public Builder setBinaryTransformSize(int binaryTransformSize) {
            this.binaryTransformSize = binaryTransformSize;
            return this;
        }

        public Builder setBinaryINd4jize(int binaryINd4jize) {
            this.binaryINd4jize = binaryINd4jize;
            return this;
        }

        public Builder setBinaryClassificationSize(int binaryClassificationSize) {
            this.binaryClassificationSize = binaryClassificationSize;
            return this;
        }

        public Builder setNumUnaryMatrices(int numUnaryMatrices) {
            this.numUnaryMatrices = numUnaryMatrices;
            return this;
        }

        public Builder setUnaryClassificationSize(int unaryClassificationSize) {
            this.unaryClassificationSize = unaryClassificationSize;
            return this;
        }

        public Builder setClassWeights(Map<Integer, Double> classWeights) {
            this.classWeights = classWeights;
            return this;
        }

        public RNTN build() {
            RNTN rt =  new RNTN(numHidden,
                    rng,
                    useINd4j,
                    combineClassification,
                    simplifiedModel,
                    randomFeatureVectors,
                    scalingForInit,
                    lowerCasefeatureNames,
                    activationFunction,
                    adagradResetFrequency,
                    regTransformINDArray,
                    featureVectors,
                    vocabCache,
                    numBinaryMatrices, binaryTransformSize, binaryINd4jize, binaryClassificationSize, numUnaryMatrices, unaryClassificationSize, classWeights);

            return rt;
        }
    }

    @Override
    public void setInputMiniBatchSize(int size){
    	this.inputMiniBatchSize = size;
    }

    @Override
    public int getInputMiniBatchSize(){
    	return inputMiniBatchSize;
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        throw new UnsupportedOperationException();
    }
}
