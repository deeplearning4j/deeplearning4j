package org.deeplearning4j.rntn;

import akka.actor.ActorSystem;
import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.FloatTensor;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.nn.learning.AdaGradFloat;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.MultiDimensionalMap;
import org.deeplearning4j.util.MultiDimensionalSet;
import org.deeplearning4j.word2vec.Word2Vec;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Recursive Neural FloatTensor Network by Socher et. al
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
public class RNTN implements Serializable {

    protected float value = 0;
    private int numOuts = 3;
    //must be same size as word vectors
    private int numHidden = 25;
    private RandomGenerator rng;
    private boolean useFloatTensors = true;
    private boolean combineClassification = true;
    private boolean simplifiedModel = true;
    private boolean randomFeatureVectors = true;
    private float scalingForInit = 1.0f;
    public final static String UNKNOWN_FEATURE = "UNK";
    private boolean lowerCasefeatureNames;
    protected ActivationFunction activationFunction = Activations.hardTanh();
    protected ActivationFunction outputActivation = Activations.softmax();
    protected AdaGradFloat paramAdaGrad;
    private int currIteration = -1;

    /** Regularization cost for the transform matrix  */
    private float regTransformMatrix = 0.001f;

    /** Regularization cost for the classification matrices */
    private float regClassification = 0.0001f;

    /** Regularization cost for the word vectors */
    private float regWordVector = 0.0001f;

    /**
     * The value to set the learning rate for each parameter when initializing adagrad.
     */
    private float initialAdagradWeight = 0.0f;

    /**
     * How many epochs between resets of the adagrad learning rates.
     * Set to 0 to never reset.
     */
    private int adagradResetFrequency = 1;

    /** Regularization cost for the transform FloatTensor  */
    private float regTransformFloatTensor = 0.001f;

    /**
     * Nx2N+1, where N is the size of the word vectors
     */
    private MultiDimensionalMap<String, String, FloatMatrix> binaryTransform;

    /**
     * 2Nx2NxN, where N is the size of the word vectors
     */
    private MultiDimensionalMap<String, String, FloatTensor> binaryFloatTensors;

    /**
     * CxN+1, where N = size of word vectors, C is the number of classes
     */
    private Map<String, FloatMatrix> unaryClassification;

    private Map<String, FloatMatrix> featureVectors;

    /**
     * CxN+1, where N = size of word vectors, C is the number of classes
     */
    private MultiDimensionalMap<String, String, FloatMatrix> binaryClassification;


    /**
     * Cached here for easy calculation of the model size;
     * MultiDimensionalMap does not return that in O(1) time
     */
    private  int numBinaryMatrices;

    /** How many elements a transformation matrix has */
    private  int binaryTransformSize;
    /** How many elements the binary transformation FloatTensors have */
    private  int binaryFloatTensorSize;
    /** How many elements a classification matrix has */
    private  int binaryClassificationSize;

    /**
     * Cached here for easy calculation of the model size;
     * MultiDimensionalMap does not return that in O(1) time
     */
    private  int numUnaryMatrices;

    /** How many elements a classification matrix has */
    private  int unaryClassificationSize;

    private FloatMatrix identity;

    private List<Tree> trainingTrees;


    private Map<Integer,Float> classWeights;

    private static Logger log = LoggerFactory.getLogger(RNTN.class);


    private ActorSystem rnTnActorSystem = ActorSystem.create("RNTN");

    private RNTN(int numHidden, RandomGenerator rng, boolean useFloatTensors, boolean combineClassification, boolean simplifiedModel, boolean randomFeatureVectors, float scalingForInit, boolean lowerCasefeatureNames, ActivationFunction activationFunction, float initialAdagradWeight, int adagradResetFrequency, float regTransformFloatTensor, Map<String, FloatMatrix> featureVectors, int numBinaryMatrices, int binaryTransformSize, int binaryFloatTensorSize, int binaryClassificationSize, int numUnaryMatrices, int unaryClassificationSize, Map<Integer, Float> classWeights) {
        this.numHidden = numHidden;
        this.rng = rng;
        this.useFloatTensors = useFloatTensors;
        this.combineClassification = combineClassification;
        this.simplifiedModel = simplifiedModel;
        this.randomFeatureVectors = randomFeatureVectors;
        this.scalingForInit = scalingForInit;
        this.lowerCasefeatureNames = lowerCasefeatureNames;
        this.activationFunction = activationFunction;
        this.initialAdagradWeight = initialAdagradWeight;
        this.adagradResetFrequency = adagradResetFrequency;
        this.regTransformFloatTensor = regTransformFloatTensor;
        this.featureVectors = featureVectors;
        this.numBinaryMatrices = numBinaryMatrices;
        this.binaryTransformSize = binaryTransformSize;
        this.binaryFloatTensorSize = binaryFloatTensorSize;
        this.binaryClassificationSize = binaryClassificationSize;
        this.numUnaryMatrices = numUnaryMatrices;
        this.unaryClassificationSize = unaryClassificationSize;
        this.classWeights = classWeights;
        init();
    }


    private void init() {

        if(rng == null)
            rng = new MersenneTwister(123);

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


        identity = FloatMatrix.eye(numHidden);

        binaryTransform = MultiDimensionalMap.newTreeBackedMap();
        binaryFloatTensors = MultiDimensionalMap.newTreeBackedMap();
        binaryClassification = MultiDimensionalMap.newTreeBackedMap();

        // When making a flat model (no symantic untying) the
        // basicCategory function will return the same basic category for
        // all labels, so all entries will map to the same matrix
        for (Pair<String, String> binary : binaryProductions) {
            String left = basicCategory(binary.getFirst());
            String right = basicCategory(binary.getSecond());
            if (binaryTransform.contains(left, right)) {
                continue;
            }

            binaryTransform.put(left, right, randomTransformMatrix());
            if (useFloatTensors) {
                binaryFloatTensors.put(left, right, randomBinaryFloatTensor());
            }

            if (!combineClassification) {
                binaryClassification.put(left, right, randomClassificationMatrix());
            }
        }

        numBinaryMatrices = binaryTransform.size();
        binaryTransformSize = numHidden * (2 * numHidden + 1);

        if (useFloatTensors) {
            binaryFloatTensorSize = numHidden * numHidden * numHidden * 4;
        } else {
            binaryFloatTensorSize = 0;
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


        featureVectors.put(UNKNOWN_FEATURE,randomWordVector());
        numUnaryMatrices = unaryClassification.size();
        unaryClassificationSize = numOuts * (numHidden + 1);
        classWeights = new HashMap<>();

    }



    void initRandomWordVectors(List<Tree> trainingTrees) {
        if (numHidden == 0) {
            throw new RuntimeException("Cannot create random word vectors for an unknown numHid");
        }

        Set<String> words = new HashSet<>();
        words.add(UNKNOWN_FEATURE);
        for (Tree tree : trainingTrees) {
            List<Tree> leaves = tree.getLeaves();
            for (Tree leaf : leaves) {
                String word = leaf.value();
                if (lowerCasefeatureNames) {
                    word = word.toLowerCase();
                }
                words.add(word);
            }
        }


        this.featureVectors = new TreeMap<>();

        for (String word : words) {
            FloatMatrix vector = randomWordVector();
            featureVectors.put(word, vector);
        }
    }




    FloatTensor randomBinaryFloatTensor() {
        float range = 1.0f / (4.0f * numHidden);
        FloatTensor floatTensor = FloatTensor.rand(numHidden * 2, numHidden * 2, numHidden, -range, range, rng);
        return floatTensor.scale(scalingForInit);
    }

    FloatMatrix randomTransformMatrix() {
        FloatMatrix binary = new FloatMatrix(numHidden, numHidden * 2 + 1);
        // bias column values are initialized zero
        FloatMatrix block = randomTransformBlock();
        binary.put(RangeUtils.interval(0,block.rows),RangeUtils.interval(0,block.columns),block);
        return SimpleBlas.scal(scalingForInit,binary);
    }

    FloatMatrix randomTransformBlock() {
        float range = 1.0f / (float) (Math.sqrt((float) numHidden) * 2.0f);
        FloatMatrix ret = FloatMatrix.randn(numHidden,numHidden);
        ret.muli(2).muli(range).subi(range).add(identity);
        return ret;
    }

    /**
     * Returns matrices of the right size for either binary or unary (terminal) classification
     */
    FloatMatrix randomClassificationMatrix() {
        // Leave the bias column with 0 values
        float range = 1.0f / (float) (Math.sqrt((float) numHidden));
        FloatMatrix ret = FloatMatrix.zeros(numOuts,numHidden + 1);
        FloatMatrix insert = FloatMatrix.randn(numOuts,numHidden);
        insert.muli(2).muli(range).subi(range);
        ret.put(RangeUtils.interval(0,insert.rows),RangeUtils.interval(0,insert.columns),insert);
        return ret;
    }

    FloatMatrix randomWordVector() {
        return randomWordVector(numHidden, rng);
    }

    static FloatMatrix randomWordVector(int size, RandomGenerator rng) {
        return MatrixUtil.uniformFloat(rng,1,size);
    }


    /**
     * Trains the network on this mini batch
     * @param trainingBatch the trees to train on
     */
    public void train(List<Tree> trainingBatch) {
        this.trainingTrees = trainingBatch;
        for(Tree t : trainingBatch) {
            forwardPropagateTree(t);
            setParameters(getParameters().subi(getValueGradient(0)));
        }
    }

    /**
     * Given a sequence of Iterators over a set of matrices, fill in all of
     * the matrices with the entries in the theta vector.  Errors are
     * thrown if the theta vector does not exactly fill the matrices.
     */
    public  void setParams(FloatMatrix theta, Iterator<? extends FloatMatrix> ... matrices) {
        int index = 0;
        for (Iterator<? extends FloatMatrix> matrixIterator : matrices) {
            while (matrixIterator.hasNext()) {
                FloatMatrix matrix = matrixIterator.next();
                int numElements = matrix.length;
                for (int i = 0; i < numElements; ++i) {
                    matrix.put(i, theta.get(index));
                    ++index;
                }
            }
        }


        if (index != theta.length) {
            throw new AssertionError("Did not entirely use the theta vector");
        }

    }

    public FloatMatrix getWForNode(Tree node) {
        if (node.children().size() == 2) {
            String leftLabel = node.children().get(0).value();
            String leftBasic = basicCategory(leftLabel);
            String rightLabel = node.children().get(1).value();
            String rightBasic = basicCategory(rightLabel);
            return binaryTransform.get(leftBasic, rightBasic);
        } else if (node.children().size() == 1) {
            throw new AssertionError("No unary transform matrices, only unary classification");
        } else {
            throw new AssertionError("Unexpected tree children size of " + node.children().size());
        }
    }

    public FloatTensor getFloatTensorForNode(Tree node) {
        if (!useFloatTensors) {
            throw new AssertionError("Not using FloatTensors");
        }
        if (node.children().size() == 2) {
            String leftLabel = node.children().get(0).value();
            String leftBasic = basicCategory(leftLabel);
            String rightLabel = node.children().get(1).value();
            String rightBasic = basicCategory(rightLabel);
            return binaryFloatTensors.get(leftBasic, rightBasic);
        } else if (node.children().size() == 1) {
            throw new AssertionError("No unary transform matrices, only unary classification");
        } else {
            throw new AssertionError("Unexpected tree children size of " + node.children().size());
        }
    }

    public FloatMatrix getClassWForNode(Tree node) {
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



    private FloatTensor getFloatTensorGradient(FloatMatrix deltaFull, FloatMatrix leftVector, FloatMatrix rightVector) {
        int size = deltaFull.length;
        FloatTensor Wt_df = new FloatTensor(size*2, size*2, size);
        FloatMatrix fullVector = FloatMatrix.concatHorizontally(leftVector, rightVector);
        for (int slice = 0; slice < size; ++slice) {
            Wt_df.setSlice(slice, SimpleBlas.scal(deltaFull.get(slice),fullVector).mmul(fullVector.transpose()));
        }
        return Wt_df;
    }



    public FloatMatrix getFeatureVector(String word) {
        FloatMatrix ret = featureVectors.get(getVocabWord(word)).transpose();
        return ret;
    }

    public String getVocabWord(String word) {
        if (lowerCasefeatureNames) {
            word = word.toLowerCase();
        }
        if (featureVectors.containsKey(word)) {
            return word;
        }
        // TODO: go through unknown words here
        return UNKNOWN_FEATURE;
    }

    public String basicCategory(String category) {
        if (simplifiedModel) {
            return "";
        }
        throw new IllegalStateException("Only simplified model enabled");
    }

    public FloatMatrix getUnaryClassification(String category) {
        category = basicCategory(category);
        return unaryClassification.get(category);
    }

    public FloatMatrix getBinaryClassification(String left, String right) {
        if (combineClassification) {
            return unaryClassification.get("");
        } else {
            left = basicCategory(left);
            right = basicCategory(right);
            return binaryClassification.get(left, right);
        }
    }

    public FloatMatrix getBinaryTransform(String left, String right) {
        left = basicCategory(left);
        right = basicCategory(right);
        return binaryTransform.get(left, right);
    }

    public FloatTensor getBinaryFloatTensor(String left, String right) {
        left = basicCategory(left);
        right = basicCategory(right);
        return binaryFloatTensors.get(left, right);
    }




    public int getNumParameters() {
        int totalSize;
        // binaryFloatTensorSize was set to 0 if useFloatTensors=false
        totalSize = numBinaryMatrices * (binaryTransform.size() + binaryClassificationSize) + binaryFloatTensorSize;
        totalSize += numUnaryMatrices * unaryClassification.size();
        totalSize += featureVectors.size() * numHidden;
        return totalSize;
    }


    public FloatMatrix getParameters() {
        return MatrixUtil.toFlattenedFloat(getNumParameters(),
                binaryTransform.values().iterator(),
                binaryClassification.values().iterator(),
                binaryFloatTensors.values().iterator(),
                unaryClassification.values().iterator(),
                featureVectors.values().iterator());
    }


    float scaleAndRegularize(MultiDimensionalMap<String, String, FloatMatrix> derivatives,
                             MultiDimensionalMap<String, String, FloatMatrix> currentMatrices,
                             float scale,
                             float regCost) {

        float cost = 0.0f; // the regularization cost
        for (MultiDimensionalMap.Entry<String, String, FloatMatrix> entry : currentMatrices.entrySet()) {
            FloatMatrix D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
            D = SimpleBlas.scal(scale,D).add(SimpleBlas.scal(regCost,entry.getValue()));
            derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
            cost += entry.getValue().mul(entry.getValue()).sum() * regCost / 2.0;
        }
        return cost;
    }

    float scaleAndRegularize(Map<String, FloatMatrix> derivatives,
                             Map<String, FloatMatrix> currentMatrices,
                             float scale,
                             float regCost) {

        float cost = 0.0f; // the regularization cost
        for (Map.Entry<String, FloatMatrix> entry : currentMatrices.entrySet()) {
            FloatMatrix D = derivatives.get(entry.getKey());
            D = SimpleBlas.scal(scale,D).add(SimpleBlas.scal(regCost,entry.getValue()));
            derivatives.put(entry.getKey(), D);
            cost += entry.getValue().mul(entry.getValue()).sum() * regCost / 2.0;
        }
        return cost;
    }

    float scaleAndRegularizeFloatTensor(MultiDimensionalMap<String, String, FloatTensor> derivatives,
                                        MultiDimensionalMap<String, String, FloatTensor> currentMatrices,
                                        float scale,
                                        float regCost) {
        float cost = 0.0f; // the regularization cost
        for (MultiDimensionalMap.Entry<String, String, FloatTensor> entry : currentMatrices.entrySet()) {
            FloatTensor D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
            D = D.scale(scale).add(entry.getValue().scale(regCost));
            derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
            cost += entry.getValue().mul(entry.getValue()).sum() * regCost / 2.0;
        }
        return cost;
    }

    private void backpropDerivativesAndError(Tree tree,
                                             MultiDimensionalMap<String, String, FloatMatrix> binaryTD,
                                             MultiDimensionalMap<String, String, FloatMatrix> binaryCD,
                                             MultiDimensionalMap<String, String, FloatTensor> binaryFloatTensorTD,
                                             Map<String, FloatMatrix> unaryCD,
                                             Map<String, FloatMatrix> wordVectorD) {
        FloatMatrix delta = new FloatMatrix(numHidden, 1);
        backpropDerivativesAndError(tree, binaryTD, binaryCD, binaryFloatTensorTD, unaryCD, wordVectorD, delta);
    }

    private void backpropDerivativesAndError(Tree tree,
                                             MultiDimensionalMap<String, String, FloatMatrix> binaryTD,
                                             MultiDimensionalMap<String, String, FloatMatrix> binaryCD,
                                             MultiDimensionalMap<String, String, FloatTensor> binaryFloatTensorTD,
                                             Map<String, FloatMatrix> unaryCD,
                                             Map<String, FloatMatrix> wordVectorD,
                                             FloatMatrix deltaUp) {
        if (tree.isLeaf()) {
            return;
        }

        FloatMatrix currentVector = tree.vector();
        String category = tree.label();
        category = basicCategory(category);

        // Build a vector that looks like 0,0,1,0,0 with an indicator for the correct class
        FloatMatrix goldLabel = new FloatMatrix(numOuts, 1);
        int goldClass = tree.goldLabel();
        if (goldClass >= 0) {
            goldLabel.put(goldClass, 1.0f);
        }

        Float nodeWeight = classWeights.get(goldClass);
        if(nodeWeight == null)
            nodeWeight = 1.0f;
        FloatMatrix predictions = tree.prediction();

        // If this is an unlabeled class, set deltaClass to 0.  We could
        // make this more efficient by eliminating various of the below
        // calculations, but this would be the easiest way to handle the
        // unlabeled class
        FloatMatrix deltaClass = goldClass >= 0 ? SimpleBlas.scal(nodeWeight,predictions.sub(goldLabel)) : new FloatMatrix(predictions.rows, predictions.columns);
        FloatMatrix localCD = deltaClass.mmul(MatrixUtil.appendBias(currentVector).transpose());

        float error = -(MatrixFunctions.log(predictions).muli(goldLabel).sum());
        error = error * nodeWeight;
        tree.setError(error);

        if (tree.isPreTerminal()) { // below us is a word vector
            unaryCD.put(category, unaryCD.get(category).add(localCD));

            String word = tree.children().get(0).label();
            word = getVocabWord(word);


            FloatMatrix currentVectorDerivative = activationFunction.apply(currentVector);
            FloatMatrix deltaFromClass = getUnaryClassification(category).transpose().mmul(deltaClass);
            deltaFromClass = deltaFromClass.get(RangeUtils.interval(0, numHidden),RangeUtils.interval(0, 1)).mul(currentVectorDerivative);
            FloatMatrix deltaFull = deltaFromClass.add(deltaUp);
            wordVectorD.put(word, wordVectorD.get(word).add(deltaFull));


        } else {
            // Otherwise, this must be a binary node
            String leftCategory = basicCategory(tree.children().get(0).label());
            String rightCategory = basicCategory(tree.children().get(1).label());
            if (combineClassification) {
                unaryCD.put("", unaryCD.get("").add(localCD));
            } else {
                binaryCD.put(leftCategory, rightCategory, binaryCD.get(leftCategory, rightCategory).add(localCD));
            }

            FloatMatrix currentVectorDerivative = activationFunction.applyDerivative(currentVector);
            FloatMatrix deltaFromClass = getBinaryClassification(leftCategory, rightCategory).transpose().mmul(deltaClass);
            //need to figure out why dimension mismatch here
            //currentVectorDerivative: 50 x 1
            //deltaFromClass 51 x 1
            FloatMatrix mult = deltaFromClass.get(RangeUtils.interval(0, numHidden),RangeUtils.interval(0, 1));
            deltaFromClass = mult.muli(currentVectorDerivative);
            FloatMatrix deltaFull = deltaFromClass.add(deltaUp);

            FloatMatrix leftVector =tree.children().get(0).vector();
            FloatMatrix rightVector = tree.children().get(1).vector();

            FloatMatrix childrenVector = MatrixUtil.appendBias(leftVector,rightVector);

            //deltaFull 50 x 1, childrenVector: 50 x 2
            FloatMatrix add = binaryTD.get(leftCategory, rightCategory);

            FloatMatrix W_df = deltaFromClass.mmul(childrenVector.transpose());
            binaryTD.put(leftCategory, rightCategory, add.add(W_df));

            FloatMatrix deltaDown;
            if (useFloatTensors) {
                FloatTensor Wt_df = getFloatTensorGradient(deltaFull, leftVector, rightVector);
                binaryFloatTensorTD.put(leftCategory, rightCategory, binaryFloatTensorTD.get(leftCategory, rightCategory).add(Wt_df));
                deltaDown = computeFloatTensorDeltaDown(deltaFull, leftVector, rightVector, getBinaryTransform(leftCategory, rightCategory), getBinaryFloatTensor(leftCategory, rightCategory));
            } else {
                deltaDown = getBinaryTransform(leftCategory, rightCategory).transpose().mmul(deltaFull);
            }

            FloatMatrix leftDerivative = activationFunction.apply(leftVector);
            FloatMatrix rightDerivative = activationFunction.apply(rightVector);
            FloatMatrix leftDeltaDown = deltaDown.get(RangeUtils.interval(0, deltaFull.rows),RangeUtils.interval( 0, 1));
            FloatMatrix rightDeltaDown = deltaDown.get(RangeUtils.interval(deltaFull.rows, deltaFull.rows * 2),RangeUtils.interval( 0, 1));
            backpropDerivativesAndError(tree.children().get(0), binaryTD, binaryCD, binaryFloatTensorTD, unaryCD, wordVectorD, leftDerivative.mul(leftDeltaDown));
            backpropDerivativesAndError(tree.children().get(1), binaryTD, binaryCD, binaryFloatTensorTD, unaryCD, wordVectorD, rightDerivative.mul(rightDeltaDown));
        }
    }

    private FloatMatrix computeFloatTensorDeltaDown(FloatMatrix deltaFull, FloatMatrix leftVector, FloatMatrix rightVector,
                                                    FloatMatrix W, FloatTensor Wt) {
        FloatMatrix WTDelta = W.transpose().mmul(deltaFull);
        FloatMatrix WTDeltaNoBias = WTDelta.get(RangeUtils.interval( 0, 1),RangeUtils.interval(0, deltaFull.rows * 2));
        int size = deltaFull.length;
        FloatMatrix deltaFloatTensor = new FloatMatrix(size * 2, 1);
        FloatMatrix fullVector = FloatMatrix.concatHorizontally(leftVector, rightVector);
        for (int slice = 0; slice < size; ++slice) {
            FloatMatrix scaledFullVector = SimpleBlas.scal(deltaFull.get(slice),fullVector);
            deltaFloatTensor = deltaFloatTensor.add(Wt.getSlice(slice).add(Wt.getSlice(slice).transpose()).mmul(scaledFullVector));
        }
        return deltaFloatTensor.add(WTDeltaNoBias);
    }


    /**
     * This is the method to call for assigning labels and node vectors
     * to the Tree.  After calling this, each of the non-leaf nodes will
     * have the node vector and the predictions of their classes
     * assigned to that subtree's node.
     */
    public void forwardPropagateTree(Tree tree) {
        FloatMatrix nodeVector;
        FloatMatrix classification;

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
            FloatMatrix wordVector = getFeatureVector(word);
            if(wordVector == null) {
                wordVector = featureVectors.get(UNKNOWN_FEATURE);
            }


            nodeVector = activationFunction.apply(wordVector);
        } else if (tree.children().size() == 1) {
            throw new AssertionError("Non-preterminal nodes of size 1 should have already been collapsed");
        } else if (tree.children().size() == 2) {
            Tree left = tree.firstChild(),right = tree.lastChild();
            forwardPropagateTree(left);
            forwardPropagateTree(right);

            String leftCategory = tree.children().get(0).label();
            String rightCategory = tree.children().get(1).label();
            FloatMatrix W = getBinaryTransform(leftCategory, rightCategory);
            classification = getBinaryClassification(leftCategory, rightCategory);

            FloatMatrix leftVector = tree.children().get(0).vector();
            FloatMatrix rightVector = tree.children().get(1).vector();

            FloatMatrix childrenVector = MatrixUtil.appendBias(leftVector,rightVector);


            if (useFloatTensors) {
                FloatTensor FloatTensor = getBinaryFloatTensor(leftCategory, rightCategory);
                FloatMatrix FloatTensorIn = FloatMatrix.concatHorizontally(leftVector, rightVector);
                FloatMatrix FloatTensorOut = FloatTensor.bilinearProducts(FloatTensorIn);
                nodeVector = activationFunction.apply(W.mmul(childrenVector).add(FloatTensorOut));
            }

            else
                nodeVector = activationFunction.apply(W.mmul(childrenVector));

        } else {
            throw new AssertionError("Tree not correctly binarized");
        }

        FloatMatrix inputWithBias  = MatrixUtil.appendBias(nodeVector);
        FloatMatrix predictions = outputActivation.apply(classification.mmul(inputWithBias));

        tree.setPrediction(predictions);
        tree.setVector(nodeVector);
    }


    /**
     * output the prediction probabilities for each tree
     * @param trees the trees to predict
     * @return the prediction probabilities for each tree
     */
    public List<FloatMatrix> output(List<Tree> trees) {
        List<FloatMatrix> ret = new ArrayList<>();
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
            ret.add(SimpleBlas.iamax(t.prediction()));
        }

        return ret;
    }

    public void setParameters(FloatMatrix params) {
        setParams(params,binaryTransform.values().iterator(),
                binaryClassification.values().iterator(),
                binaryFloatTensors.values().iterator(),
                unaryClassification.values().iterator(),
                featureVectors.values().iterator());
    }




    public FloatMatrix getValueGradient(int iterations) {


        // We use TreeMap for each of these so that they stay in a
        // canonical sorted order
        // TODO: factor out the initialization routines
        // binaryTD stands for Transform Derivatives
        final MultiDimensionalMap<String, String, FloatMatrix> binaryTD = MultiDimensionalMap.newTreeBackedMap();
        // the derivatives of the FloatTensors for the binary nodes
        final MultiDimensionalMap<String, String, FloatTensor> binaryFloatTensorTD = MultiDimensionalMap.newTreeBackedMap();
        // binaryCD stands for Classification Derivatives
        final MultiDimensionalMap<String, String, FloatMatrix> binaryCD = MultiDimensionalMap.newTreeBackedMap();

        // unaryCD stands for Classification Derivatives
        final Map<String, FloatMatrix> unaryCD = new TreeMap<>();

        // word vector derivatives
        final Map<String, FloatMatrix> wordVectorD = new TreeMap<>();

        for (MultiDimensionalMap.Entry<String, String, FloatMatrix> entry : binaryTransform.entrySet()) {
            int numRows = entry.getValue().rows;
            int numCols = entry.getValue().columns;

            binaryTD.put(entry.getFirstKey(), entry.getSecondKey(), new FloatMatrix(numRows, numCols));
        }

        if (!combineClassification) {
            for (MultiDimensionalMap.Entry<String, String, FloatMatrix> entry :  binaryClassification.entrySet()) {
                int numRows = entry.getValue().rows;
                int numCols = entry.getValue().columns;

                binaryCD.put(entry.getFirstKey(), entry.getSecondKey(), new FloatMatrix(numRows, numCols));
            }
        }

        if (useFloatTensors) {
            for (MultiDimensionalMap.Entry<String, String, FloatTensor> entry : binaryFloatTensors.entrySet()) {
                int numRows = entry.getValue().rows();
                int numCols = entry.getValue().columns;
                int numSlices = entry.getValue().slices();

                binaryFloatTensorTD.put(entry.getFirstKey(), entry.getSecondKey(), new FloatTensor(numRows, numCols, numSlices));
            }
        }

        for (Map.Entry<String, FloatMatrix> entry : unaryClassification.entrySet()) {
            int numRows = entry.getValue().rows;
            int numCols = entry.getValue().columns;
            unaryCD.put(entry.getKey(), new FloatMatrix(numRows, numCols));
        }
        for (Map.Entry<String, FloatMatrix> entry : featureVectors.entrySet()) {
            int numRows = entry.getValue().rows;
            int numCols = entry.getValue().columns;
            wordVectorD.put(entry.getKey(), new FloatMatrix(numRows, numCols));
        }


        final List<Tree> forwardPropTrees = new CopyOnWriteArrayList<>();
        Parallelization.iterateInParallel(trainingTrees,new Parallelization.RunnableWithParams<Tree>() {

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
        Parallelization.iterateInParallel(forwardPropTrees,new Parallelization.RunnableWithParams<Tree>() {

            public void run(Tree currentItem, Object[] args) {
                backpropDerivativesAndError(currentItem, binaryTD, binaryCD, binaryFloatTensorTD, unaryCD, wordVectorD);
                error.addAndGet(currentItem.errorSum());

            }
        },new Parallelization.RunnableWithParams<Tree>() {

            public void run(Tree currentItem, Object[] args) {
            }
        },rnTnActorSystem,new Object[]{ binaryTD, binaryCD, binaryFloatTensorTD, unaryCD, wordVectorD});



        // scale the error by the number of sentences so that the
        // regularization isn't drowned out for large training batchs
        float scale = (1.0f / trainingTrees.size());
        value = error.floatValue() * scale;

        value += scaleAndRegularize(binaryTD, binaryTransform, scale, regTransformMatrix);
        value += scaleAndRegularize(binaryCD, binaryClassification, scale, regClassification);
        value += scaleAndRegularizeFloatTensor(binaryFloatTensorTD, binaryFloatTensors, scale, regTransformFloatTensor);
        value += scaleAndRegularize(unaryCD, unaryClassification, scale, regClassification);
        value += scaleAndRegularize(wordVectorD, featureVectors, scale, regWordVector);

        FloatMatrix derivative = MatrixUtil.toFlattenedFloat(getNumParameters(), binaryTD.values().iterator(), binaryCD.values().iterator(), binaryFloatTensorTD.values().iterator()
                , unaryCD.values().iterator(), wordVectorD.values().iterator());

        if(paramAdaGrad == null)
            paramAdaGrad = new AdaGradFloat(1,derivative.columns);

        derivative.muli(paramAdaGrad.getLearningRates(derivative));

        return derivative;
    }


    public float getValue() {
        return value;
    }

    public static class Builder {
        //must be same size as word vectors
        private int numHidden;
        private RandomGenerator rng;
        private boolean useFloatTensors;
        private boolean combineClassification = true;
        private boolean simplifiedModel = true;
        private boolean randomFeatureVectors;
        private float scalingForInit = 1e-3f;
        private boolean lowerCasefeatureNames;
        private ActivationFunction activationFunction = Activations.sigmoid(),
                outputActivationFunction = Activations.softmax();
        private float initialAdagradWeight;
        private int adagradResetFrequency;
        private float regTransformFloatTensor;
        private Map<String, FloatMatrix> featureVectors;
        private int numBinaryMatrices;
        private int binaryTransformSize;
        private int binaryFloatTensorSize;
        private int binaryClassificationSize;
        private int numUnaryMatrices;
        private int unaryClassificationSize;
        private Map<Integer, Float> classWeights;


        public Builder withOutputActivation(ActivationFunction outputActivationFunction) {
            this.outputActivationFunction = outputActivationFunction;
            return this;
        }

        public Builder setFeatureVectors(Word2Vec vec) {
            setFeatureVectors(vec.toVocabFloat());
            return this;
        }

        public Builder setNumHidden(int numHidden) {
            this.numHidden = numHidden;
            return this;
        }

        public Builder setRng(RandomGenerator rng) {
            this.rng = rng;
            return this;
        }

        public Builder setUseTensors(boolean useFloatTensors) {
            this.useFloatTensors = useFloatTensors;
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

        public Builder setScalingForInit(float scalingForInit) {
            this.scalingForInit = scalingForInit;
            return this;
        }

        public Builder setLowerCasefeatureNames(boolean lowerCasefeatureNames) {
            this.lowerCasefeatureNames = lowerCasefeatureNames;
            return this;
        }

        public Builder setActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder setInitialAdagradWeight(float initialAdagradWeight) {
            this.initialAdagradWeight = initialAdagradWeight;
            return this;
        }

        public Builder setAdagradResetFrequency(int adagradResetFrequency) {
            this.adagradResetFrequency = adagradResetFrequency;
            return this;
        }

        public Builder setRegTransformFloatTensor(float regTransformFloatTensor) {
            this.regTransformFloatTensor = regTransformFloatTensor;
            return this;
        }

        public Builder setFeatureVectors(Map<String, FloatMatrix> featureVectors) {
            this.featureVectors = featureVectors;
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

        public Builder setBinaryFloatTensorSize(int binaryFloatTensorSize) {
            this.binaryFloatTensorSize = binaryFloatTensorSize;
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

        public Builder setClassWeights(Map<Integer, Float> classWeights) {
            this.classWeights = classWeights;
            return this;
        }

        public RNTN build() {
            return new RNTN(numHidden, rng, useFloatTensors, combineClassification, simplifiedModel, randomFeatureVectors, scalingForInit, lowerCasefeatureNames, activationFunction, initialAdagradWeight, adagradResetFrequency, regTransformFloatTensor, featureVectors, numBinaryMatrices, binaryTransformSize, binaryFloatTensorSize, binaryClassificationSize, numUnaryMatrices, unaryClassificationSize, classWeights);
        }
    }





}
