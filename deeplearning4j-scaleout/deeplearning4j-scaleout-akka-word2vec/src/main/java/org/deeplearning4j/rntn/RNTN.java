package org.deeplearning4j.rntn;

import static org.deeplearning4j.linalg.indexing.NDArrayIndex.interval;

import akka.actor.ActorSystem;
import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;

import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.util.MultiDimensionalMap;
import org.deeplearning4j.util.MultiDimensionalSet;
import org.deeplearning4j.word2vec.Word2Vec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Recursive Neural INDArray Network by Socher et. al
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
    private boolean useINDArrays = true;
    private boolean combineClassification = true;
    private boolean simplifiedModel = true;
    private boolean randomFeatureVectors = true;
    private float scalingForInit = 1.0f;
    public final static String UNKNOWN_FEATURE = "UNK";
    private boolean lowerCasefeatureNames;
    protected ActivationFunction activationFunction = Activations.tanh();
    protected ActivationFunction outputActivation = Activations.softMaxRows();
    protected AdaGrad paramAdaGrad;

    /** Regularization cost for the applyTransformToOrigin matrix  */
    private float regTransformMatrix = 0.001f;

    /** Regularization cost for the classification matrices */
    private float regClassification = 0.0001f;

    /** Regularization cost for the word vectors */
    private float regWordVector = 0.0001f;


    /**
     * How many epochs between resets of the adagrad learning rates.
     * Set to 0 to never reset.
     */
    private int adagradResetFrequency = 1;

    /** Regularization cost for the applyTransformToOrigin INDArray  */
    private float regTransformINDArray = 0.001f;

    /**
     * Nx2N+1, where N is the size of the word vectors
     */
    private MultiDimensionalMap<String, String, INDArray> binaryTransform;

    /**
     * 2Nx2NxN, where N is the size of the word vectors
     */
    private MultiDimensionalMap<String, String, INDArray> binaryINDArrays;

    /**
     * CxN+1, where N = size of word vectors, C is the number of classes
     */
    private Map<String, INDArray> unaryClassification;

    private Map<String, INDArray> featureVectors;

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
    /** How many elements the binary transformation INDArrays have */
    private  int binaryINDArraySize;
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

    private List<Tree> trainingTrees;


    private Map<Integer,Float> classWeights;

    private static Logger log = LoggerFactory.getLogger(RNTN.class);


    private transient ActorSystem rnTnActorSystem = ActorSystem.create("RNTN");

    private RNTN(int numHidden, RandomGenerator rng, boolean useINDArrays, boolean combineClassification, boolean simplifiedModel, boolean randomFeatureVectors, float scalingForInit, boolean lowerCasefeatureNames, ActivationFunction activationFunction, int adagradResetFrequency, float regTransformINDArray, Map<String, INDArray> featureVectors, int numBinaryMatrices, int binaryTransformSize, int binaryINDArraySize, int binaryClassificationSize, int numUnaryMatrices, int unaryClassificationSize, Map<Integer, Float> classWeights) {
        this.numHidden = numHidden;
        this.rng = rng;
        this.useINDArrays = useINDArrays;
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
        this.binaryINDArraySize = binaryINDArraySize;
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


        identity = NDArrays.eye(numHidden);

        binaryTransform = MultiDimensionalMap.newTreeBackedMap();
        binaryINDArrays = MultiDimensionalMap.newTreeBackedMap();
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
            if (useINDArrays) {
                binaryINDArrays.put(left, right, randomBinaryINDArray());
            }

            if (!combineClassification) {
                binaryClassification.put(left, right, randomClassificationMatrix());
            }
        }

        numBinaryMatrices = binaryTransform.size();
        binaryTransformSize = numHidden * (2 * numHidden + 1);

        if (useINDArrays) {
            binaryINDArraySize = numHidden * numHidden * numHidden * 4;
        } else {
            binaryINDArraySize = 0;
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




    INDArray randomBinaryINDArray() {
        float range = 1.0f / (4.0f * numHidden);
        INDArray ret = NDArrays.rand(new int[]{numHidden * 2, numHidden * 2, numHidden}, -range, range, rng);
        return ret.muli(scalingForInit);
    }

    INDArray randomTransformMatrix() {
        INDArray binary = NDArrays.create(numHidden, numHidden * 2 + 1);
        // bias column values are initialized zero
        INDArray block = randomTransformBlock();
        binary.put(new NDArrayIndex[] {interval(0,block.rows()),interval(0,block.columns())},block);
        binary.put(new NDArrayIndex[]{interval(0,block.rows()),interval(numHidden,numHidden + block.columns())},randomTransformBlock());
        return NDArrays.getBlasWrapper().scal(scalingForInit,binary);
    }

    INDArray randomTransformBlock() {
        float range = 1.0f / (float) (Math.sqrt((float) numHidden) * 2.0f);
        INDArray ret = NDArrays.rand(numHidden,numHidden,-range,range,rng).add(identity);
        return ret;
    }

    /**
     * Returns matrices of the right size for either binary or unary (terminal) classification
     */
    INDArray randomClassificationMatrix() {
        // Leave the bias column with 0 values
        float range = 1.0f / (float) (Math.sqrt((float) numHidden));
        INDArray ret = NDArrays.zeros(numOuts,numHidden + 1);
        INDArray insert = NDArrays.rand(numOuts,numHidden,-range,range,rng);
        ret.put(new NDArrayIndex[] {interval(0,numOuts),interval(0,numHidden)},insert);
        return NDArrays.getBlasWrapper().scal(scalingForInit,ret);
    }

    INDArray randomWordVector() {
        return NDArrays.rand(numHidden,1, rng);
    }




    /**
     * Trains the network on this mini batch
     * @param trainingBatch the trees to iterate on
     */
    public void train(List<Tree> trainingBatch) {
        this.trainingTrees = trainingBatch;
        for(Tree t : trainingBatch) {
            forwardPropagateTree(t);
            setParameters(getParameters().subi(getValueGradient(0)));
        }
    }

    /**
     * Given a sequence of Iterators over a applyTransformToDestination of matrices, fill in all of
     * the matrices with the entries in the theta vector.  Errors are
     * thrown if the theta vector does not exactly fill the matrices.
     */
    public  void setParams(INDArray theta, Iterator<? extends INDArray> ... matrices) {
        int index = 0;
        for (Iterator<? extends INDArray> matrixIterator : matrices) {
            while (matrixIterator.hasNext()) {
                INDArray matrix = matrixIterator.next();
                for (int i = 0; i < matrix.length(); ++i) {
                    matrix.put(i, theta.getScalar(index));
                    ++index;
                }
            }
        }


        if (index != theta.length()) {
            throw new AssertionError("Did not entirely use the theta vector");
        }

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
        if (!useINDArrays) {
            throw new AssertionError("Not using INDArrays");
        }
        if (node.children().size() == 2) {
            String leftLabel = node.children().get(0).value();
            String leftBasic = basicCategory(leftLabel);
            String rightLabel = node.children().get(1).value();
            String rightBasic = basicCategory(rightLabel);
            return binaryINDArrays.get(leftBasic, rightBasic);
        } else if (node.children().size() == 1) {
            throw new AssertionError("No unary applyTransformToOrigin matrices, only unary classification");
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
        INDArray Wt_df = NDArrays.create(new int[]{size*2, size*2, size});
        INDArray fullVector = NDArrays.concatHorizontally(leftVector, rightVector);
        for (int slice = 0; slice < size; ++slice) {
            Wt_df.putSlice(slice, NDArrays.getBlasWrapper().scal((float) deltaFull.getScalar(slice).element(),fullVector).mmul(fullVector.transpose()));
        }
        return Wt_df;
    }



    public INDArray getFeatureVector(String word) {
        INDArray ret = featureVectors.get(getVocabWord(word)).transpose();
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
        return binaryINDArrays.get(left, right);
    }




    public int getNumParameters() {
        int totalSize;
        // binaryINDArraySize was applyTransformToDestination to 0 if useINDArrays=false
        totalSize = numBinaryMatrices * (binaryTransform.size() + binaryClassificationSize) + binaryINDArraySize;
        totalSize += numUnaryMatrices * unaryClassification.size();
        totalSize += featureVectors.size() * numHidden;
        return totalSize;
    }


    public INDArray getParameters() {
        return NDArrays.toFlattened(getNumParameters(),
                binaryTransform.values().iterator(),
                binaryClassification.values().iterator(),
                binaryINDArrays.values().iterator(),
                unaryClassification.values().iterator(),
                featureVectors.values().iterator());
    }


    float scaleAndRegularize(MultiDimensionalMap<String, String, INDArray> derivatives,
                             MultiDimensionalMap<String, String, INDArray> currentMatrices,
                             float scale,
                             float regCost) {

        float cost = 0.0f; // the regularization cost
        for (MultiDimensionalMap.Entry<String, String, INDArray> entry : currentMatrices.entrySet()) {
            INDArray D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
            D = NDArrays.getBlasWrapper().scal(scale,D).add(NDArrays.getBlasWrapper().scal(regCost,entry.getValue()));
            derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
            cost += (double) entry.getValue().mul(entry.getValue()).sum(Integer.MAX_VALUE).element() * regCost / 2.0;
        }
        return cost;
    }

    float scaleAndRegularize(Map<String, INDArray> derivatives,
                             Map<String, INDArray> currentMatrices,
                             float scale,
                             float regCost) {

        float cost = 0.0f; // the regularization cost
        for (Map.Entry<String, INDArray> entry : currentMatrices.entrySet()) {
            INDArray D = derivatives.get(entry.getKey());
            D = NDArrays.getBlasWrapper().scal(scale,D).add(NDArrays.getBlasWrapper().scal(regCost,entry.getValue()));
            derivatives.put(entry.getKey(), D);
            cost += (double) entry.getValue().mul(entry.getValue()).sum(Integer.MAX_VALUE).element() * regCost / 2.0;
        }
        return cost;
    }

    float scaleAndRegularizeINDArray(MultiDimensionalMap<String, String, INDArray> derivatives,
                                        MultiDimensionalMap<String, String, INDArray> currentMatrices,
                                        float scale,
                                        float regCost) {
        float cost = 0.0f; // the regularization cost
        for (MultiDimensionalMap.Entry<String, String, INDArray> entry : currentMatrices.entrySet()) {
            INDArray D = derivatives.get(entry.getFirstKey(), entry.getSecondKey());
            D = D.muli(scale).add(entry.getValue().muli(regCost));
            derivatives.put(entry.getFirstKey(), entry.getSecondKey(), D);
            cost += (double)  entry.getValue().mul(entry.getValue()).sum(Integer.MAX_VALUE).element() * regCost / 2.0;
        }
        return cost;
    }

    private void backpropDerivativesAndError(Tree tree,
                                             MultiDimensionalMap<String, String, INDArray> binaryTD,
                                             MultiDimensionalMap<String, String, INDArray> binaryCD,
                                             MultiDimensionalMap<String, String, INDArray> binaryINDArrayTD,
                                             Map<String, INDArray> unaryCD,
                                             Map<String, INDArray> wordVectorD) {
        INDArray delta = NDArrays.create(numHidden, 1);
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
        INDArray goldLabel = NDArrays.create(numOuts, 1);
        int goldClass = tree.goldLabel();
        if (goldClass >= 0) {
            goldLabel.putScalar(goldClass, 1.0f);
        }

        Float nodeWeight = classWeights.get(goldClass);
        if(nodeWeight == null)
            nodeWeight = 1.0f;
        INDArray predictions = tree.prediction();

        // If this is an unlabeled class, applyTransformToDestination deltaClass to 0.  We could
        // make this more efficient by eliminating various of the below
        // calculations, but this would be the easiest way to handle the
        // unlabeled class
        INDArray deltaClass = goldClass >= 0 ? NDArrays.getBlasWrapper().scal(nodeWeight,predictions.sub(goldLabel)) : NDArrays.create(predictions.rows(), predictions.columns());
        INDArray localCD = deltaClass.mmul(NDArrays.appendBias(currentVector).transpose());

        float error = -(float) (Transforms.log(predictions).muli(goldLabel).sum(Integer.MAX_VALUE).element());
        error = error * nodeWeight;
        tree.setError(error);

        if (tree.isPreTerminal()) { // below us is a word vector
            unaryCD.put(category, unaryCD.get(category).add(localCD));

            String word = tree.children().get(0).label();
            word = getVocabWord(word);


            INDArray currentVectorDerivative = activationFunction.apply(currentVector);
            INDArray deltaFromClass = getUnaryClassification(category).transpose().mmul(deltaClass);
            deltaFromClass = deltaFromClass.get(interval(0, numHidden),interval(0, 1)).mul(currentVectorDerivative);
            INDArray deltaFull = deltaFromClass.add(deltaUp);
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

            INDArray currentVectorDerivative = activationFunction.applyDerivative(currentVector);
            INDArray deltaFromClass = getBinaryClassification(leftCategory, rightCategory).transpose().mmul(deltaClass);

            INDArray mult = deltaFromClass.get(interval(0, numHidden),interval(0, 1));
            deltaFromClass = mult.muli(currentVectorDerivative);
            INDArray deltaFull = deltaFromClass.add(deltaUp);

            INDArray leftVector =tree.children().get(0).vector();
            INDArray rightVector = tree.children().get(1).vector();

            INDArray childrenVector = NDArrays.appendBias(leftVector,rightVector);

            //deltaFull 50 x 1, childrenVector: 50 x 2
            INDArray add = binaryTD.get(leftCategory, rightCategory);

            INDArray W_df = deltaFromClass.mmul(childrenVector.transpose());
            binaryTD.put(leftCategory, rightCategory, add.add(W_df));

            INDArray deltaDown;
            if (useINDArrays) {
                INDArray Wt_df = getINDArrayGradient(deltaFull, leftVector, rightVector);
                binaryINDArrayTD.put(leftCategory, rightCategory, binaryINDArrayTD.get(leftCategory, rightCategory).add(Wt_df));
                deltaDown = computeINDArrayDeltaDown(deltaFull, leftVector, rightVector, getBinaryTransform(leftCategory, rightCategory), getBinaryINDArray(leftCategory, rightCategory));
            } else {
                deltaDown = getBinaryTransform(leftCategory, rightCategory).transpose().mmul(deltaFull);
            }

            INDArray leftDerivative = activationFunction.apply(leftVector);
            INDArray rightDerivative = activationFunction.apply(rightVector);
            INDArray leftDeltaDown = deltaDown.get(interval(0, deltaFull.rows()),interval( 0, 1));
            INDArray rightDeltaDown = deltaDown.get(interval(deltaFull.rows(), deltaFull.rows() * 2),interval( 0, 1));
            backpropDerivativesAndError(tree.children().get(0), binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD, leftDerivative.mul(leftDeltaDown));
            backpropDerivativesAndError(tree.children().get(1), binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD, rightDerivative.mul(rightDeltaDown));
        }
    }

    private INDArray computeINDArrayDeltaDown(INDArray deltaFull, INDArray leftVector, INDArray rightVector,
                                                INDArray W, INDArray Wt) {
        INDArray WTDelta = W.transpose().mmul(deltaFull);
        INDArray WTDeltaNoBias = WTDelta.get(interval( 0, 1),interval(0, deltaFull.rows() * 2));
        int size = deltaFull.length();
        INDArray deltaINDArray = NDArrays.create(size * 2, 1);
        INDArray fullVector = NDArrays.concatHorizontally(leftVector, rightVector);
        for (int slice = 0; slice < size; ++slice) {
            INDArray scaledFullVector = NDArrays.getBlasWrapper().scal((float) deltaFull.getScalar(slice).element(),fullVector);
            deltaINDArray = deltaINDArray.add(Wt.slice(slice).add(Wt.slice(slice).transpose()).mmul(scaledFullVector));
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
            INDArray W = getBinaryTransform(leftCategory, rightCategory);
            classification = getBinaryClassification(leftCategory, rightCategory);

            INDArray leftVector = tree.children().get(0).vector();
            INDArray rightVector = tree.children().get(1).vector();

            INDArray childrenVector = NDArrays.appendBias(leftVector,rightVector);


            if (useINDArrays) {
                INDArray floatT = getBinaryINDArray(leftCategory, rightCategory);
                INDArray INDArrayIn = NDArrays.concatHorizontally(leftVector, rightVector);
                INDArray INDArrayOut = NDArrays.bilinearProducts(floatT,INDArrayIn);
                nodeVector = activationFunction.apply(W.mmul(childrenVector).add(INDArrayOut));
            }

            else
                nodeVector = activationFunction.apply(W.mmul(childrenVector));

        } else {
            throw new AssertionError("Tree not correctly binarized");
        }

        INDArray inputWithBias  = NDArrays.appendBias(nodeVector);
        INDArray preAct = classification.mmul(inputWithBias);
        INDArray predictions = outputActivation.apply(preAct);

        tree.setPrediction(predictions);
        tree.setVector(nodeVector);
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
            ret.add(NDArrays.getBlasWrapper().iamax(t.prediction()));
        }

        return ret;
    }

    public void setParameters(INDArray params) {
        setParams(params,binaryTransform.values().iterator(),
                binaryClassification.values().iterator(),
                binaryINDArrays.values().iterator(),
                unaryClassification.values().iterator(),
                featureVectors.values().iterator());
    }




    public INDArray getValueGradient(int iterations) {


        // We use TreeMap for each of these so that they stay in a
        // canonical sorted order
        // TODO: factor out the initialization routines
        // binaryTD stands for Transform Derivatives
        final MultiDimensionalMap<String, String, INDArray> binaryTD = MultiDimensionalMap.newTreeBackedMap();
        // the derivatives of the INDArrays for the binary nodes
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

            binaryTD.put(entry.getFirstKey(), entry.getSecondKey(), NDArrays.create(numRows, numCols));
        }

        if (!combineClassification) {
            for (MultiDimensionalMap.Entry<String, String, INDArray> entry :  binaryClassification.entrySet()) {
                int numRows = entry.getValue().rows();
                int numCols = entry.getValue().columns();

                binaryCD.put(entry.getFirstKey(), entry.getSecondKey(), NDArrays.create(numRows, numCols));
            }
        }

        if (useINDArrays) {
            for (MultiDimensionalMap.Entry<String, String, INDArray> entry : binaryINDArrays.entrySet()) {
                int numRows = entry.getValue().rows();
                int numCols = entry.getValue().columns();
                int numSlices = entry.getValue().slices();

                binaryINDArrayTD.put(entry.getFirstKey(), entry.getSecondKey(), NDArrays.create(new int[]{numRows, numCols, numSlices}));
            }
        }

        for (Map.Entry<String, INDArray> entry : unaryClassification.entrySet()) {
            int numRows = entry.getValue().rows();
            int numCols = entry.getValue().columns();
            unaryCD.put(entry.getKey(), NDArrays.create(numRows, numCols));
        }
        for (Map.Entry<String, INDArray> entry : featureVectors.entrySet()) {
            int numRows = entry.getValue().rows();
            int numCols = entry.getValue().columns();
            wordVectorD.put(entry.getKey(), NDArrays.create(numRows, numCols));
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
                backpropDerivativesAndError(currentItem, binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD);
                error.addAndGet(currentItem.errorSum());

            }
        },new Parallelization.RunnableWithParams<Tree>() {

            public void run(Tree currentItem, Object[] args) {
            }
        },rnTnActorSystem,new Object[]{ binaryTD, binaryCD, binaryINDArrayTD, unaryCD, wordVectorD});



        // scale the error by the number of sentences so that the
        // regularization isn't drowned out for large training batchs
        float scale = (1.0f / trainingTrees.size());
        value = error.floatValue() * scale;

        value += scaleAndRegularize(binaryTD, binaryTransform, scale, regTransformMatrix);
        value += scaleAndRegularize(binaryCD, binaryClassification, scale, regClassification);
        value += scaleAndRegularizeINDArray(binaryINDArrayTD, binaryINDArrays, scale, regTransformINDArray);
        value += scaleAndRegularize(unaryCD, unaryClassification, scale, regClassification);
        value += scaleAndRegularize(wordVectorD, featureVectors, scale, regWordVector);

        INDArray derivative = NDArrays.toFlattened(getNumParameters(), binaryTD.values().iterator(), binaryCD.values().iterator(), binaryINDArrayTD.values().iterator()
                , unaryCD.values().iterator(), wordVectorD.values().iterator());

        if(paramAdaGrad == null)
            paramAdaGrad = new AdaGrad(1,derivative.columns());

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
        private boolean useINDArrays;
        private boolean combineClassification = true;
        private boolean simplifiedModel = true;
        private boolean randomFeatureVectors;
        private float scalingForInit = 1e-3f;
        private boolean lowerCasefeatureNames;
        private ActivationFunction activationFunction = Activations.sigmoid(),
                outputActivationFunction = Activations.softmax();
        private int adagradResetFrequency;
        private float regTransformINDArray;
        private Map<String, INDArray> featureVectors;
        private int numBinaryMatrices;
        private int binaryTransformSize;
        private int binaryINDArraySize;
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

        public Builder setUseTensors(boolean useINDArrays) {
            this.useINDArrays = useINDArrays;
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



        public Builder setAdagradResetFrequency(int adagradResetFrequency) {
            this.adagradResetFrequency = adagradResetFrequency;
            return this;
        }

        public Builder setRegTransformINDArray(float regTransformINDArray) {
            this.regTransformINDArray = regTransformINDArray;
            return this;
        }

        public Builder setFeatureVectors(Map<String, INDArray> featureVectors) {
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

        public Builder setBinaryINDArraySize(int binaryINDArraySize) {
            this.binaryINDArraySize = binaryINDArraySize;
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
            return new RNTN(numHidden, rng, useINDArrays, combineClassification, simplifiedModel, randomFeatureVectors, scalingForInit, lowerCasefeatureNames, activationFunction, adagradResetFrequency, regTransformINDArray, featureVectors, numBinaryMatrices, binaryTransformSize, binaryINDArraySize, binaryClassificationSize, numUnaryMatrices, unaryClassificationSize, classWeights);
        }
    }





}
