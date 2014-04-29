package org.deeplearning4j.rntn;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.util.MultiDimensionalMap;
import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.Map;

/**
 * Recursive Neural Tensor Network by Socher et. al
 *
 */
public class RNTN implements Serializable {

    private Map<String,DoubleMatrix> featureVectors;
    private int numOuts;
    private int numHidden;
    private RandomGenerator rng;

    /**
     * Nx2N+1, where N is the size of the word vectors
     */
    public MultiDimensionalMap<String, String, DoubleMatrix> binaryTransform;

    /**
     * 2Nx2NxN, where N is the size of the word vectors
     */
    public MultiDimensionalMap<String, String, DoubleMatrix> binaryTensors;

    /**
     * CxN+1, where N = size of word vectors, C is the number of classes
     */
    public MultiDimensionalMap<String, String, DoubleMatrix> binaryClassification;


    /**
     * Cached here for easy calculation of the model size;
     * TwoDimensionalMap does not return that in O(1) time
     */
    private  int numBinaryMatrices;

    /** How many elements a transformation matrix has */
    private  int binaryTransformSize;
    /** How many elements the binary transformation tensors have */
    private  int binaryTensorSize;
    /** How many elements a classification matrix has */
    private  int binaryClassificationSize;

    /**
     * Cached here for easy calculation of the model size;
     * TwoDimensionalMap does not return that in O(1) time
     */
    private  int numUnaryMatrices;

    /** How many elements a classification matrix has */
    private  int unaryClassificationSize;




}
