package org.nd4j.linalg.dataset.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**MultiDataSet is an interface for representing complex data sets, that have (potentially) multiple inputs and outputs
 * For example, some complex neural network architectures may have multiple independent inputs, and multiple independent
 * outputs. These inputs and outputs need not even be the same type of data: for example, images in and sequences out, etc
 *
 */
public interface MultiDataSet extends Serializable {

    /** Number of arrays of features/input data in the MultiDataSet */
    int numFeatureArrays();

    /** Number of arrays of label/output data in the MultiDataSet */
    int numLabelsArrays();

    /** Get all of the input features, as an array of INDArrays */
    INDArray[] getFeatures();

    /** Get a single feature/input array */
    INDArray getFeatures(int index);

    /** Set all of the features arrays for the MultiDataSet */
    void setFeatures(INDArray[] features);

    /** Set a single features array (by index) for the MultiDataSet */
    void setFeatures(int idx, INDArray features);

    /** Get all of the labels, as an array of INDArrays */
    INDArray[] getLabels();

    /** Get a single label/output array */
    INDArray getLabels(int index);

    /** Set all of the labels arrays for the MultiDataSet */
    void setLabels(INDArray[] labels);

    /** Set a single labels array (by index) for the MultiDataSet */
    void setLabels(int idx, INDArray labels);

    /** Whether there are any mask arrays (features or labels) present for this MultiDataSet */
    boolean hasMaskArrays();

    /** Get the feature mask arrays. May return null if no feature mask arrays are present; otherwise,
     * any entry in the array may be null if no mask array is present for that particular feature
     */
    INDArray[] getFeaturesMaskArrays();

    /** Get the specified feature mask array. Returns null if no feature mask is present for that particular feature/input */
    INDArray getFeaturesMaskArray(int index);

    /** Set the feature mask arrays */
    void setFeaturesMaskArrays(INDArray[] maskArrays);

    /** Set a single feature mask array by index */
    void setFeaturesMaskArray(int idx, INDArray maskArray);

    /** Get the labels mask arrays. May return null if no labels mask arrays are present; otherwise,
     * any entry in the array may be null if no mask array is present for that particular label
     */
    INDArray[] getLabelsMaskArrays();

    /** Get the specified label mask array. Returns null if no label mask is present for that particular feature/input */
    INDArray getLabelsMaskArray(int index);

    /** Set the labels mask arrays */
    void setLabelsMaskArray(INDArray[] labels);

    /** Set a single labels mask array by index */
    void setLabelsMaskArray(int idx, INDArray labelsMaskArray);

}
