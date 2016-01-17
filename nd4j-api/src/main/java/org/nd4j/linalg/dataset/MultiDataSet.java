package org.nd4j.linalg.dataset;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**Implementation of {@link org.nd4j.linalg.dataset.api.MultiDataSet}
 * @author Alex Black
 */
public class MultiDataSet implements org.nd4j.linalg.dataset.api.MultiDataSet {

    private INDArray[] features;
    private INDArray[] labels;
    private INDArray[] featuresMaskArrays;
    private INDArray[] labelsMaskArrays;

    /** MultiDataSet constructor with single features/labels input, no mask arrays */
    public MultiDataSet(INDArray features, INDArray labels){
        this(new INDArray[]{features}, new INDArray[]{labels});
    }

    /** MultiDataSet constructor with no mask arrays */
    public MultiDataSet(INDArray[] features, INDArray[] labels){
        this(features,labels,null,null);
    }

    /**
     *
     * @param features The features (inputs) to the algorithm/neural network
     * @param labels The labels (outputs) to the algorithm/neural network
     * @param featuresMaskArrays The mask arrays for the features. May be null. Typically used with variable-length time series models, etc
     * @param labelsMaskArrays The mask arrays for the labels. May be null. Typically used with variable-length time series models, etc
     */
    public MultiDataSet(INDArray[] features, INDArray[] labels, INDArray[] featuresMaskArrays, INDArray[] labelsMaskArrays ){
        if(features != null && featuresMaskArrays != null){
            if(features.length != featuresMaskArrays.length) throw new IllegalArgumentException("Invalid features / features mask arrays combination: "
                    + "features and features mask arrays must not be different lengths");
        }
        if(labels != null && labelsMaskArrays != null ){
            if(labels.length != labelsMaskArrays.length) throw new IllegalArgumentException("Invalid labels / labels mask arrays combination: "
                    + "labels and labels mask arrays must not be different lengths");
        }

        this.features = features;
        this.labels = labels;
        this.featuresMaskArrays = featuresMaskArrays;
        this.labelsMaskArrays = labelsMaskArrays;
    }


    @Override
    public int numFeatureArrays() {
        return (features != null ? features.length : 0);
    }

    @Override
    public int numLabelsArrays() {
        return (labels != null ? labels.length : 0);
    }

    @Override
    public INDArray[] getFeatures() {
        return features;
    }

    @Override
    public INDArray getFeatures(int index) {
        return features[index];
    }

    @Override
    public void setFeatures(INDArray[] features) {
        this.features = features;
    }

    @Override
    public void setFeatures(int idx, INDArray features) {
        this.features[idx] = features;
    }

    @Override
    public INDArray[] getLabels() {
        return labels;
    }

    @Override
    public INDArray getLabels(int index) {
        return labels[index];
    }

    @Override
    public void setLabels(INDArray[] labels) {
        this.labels = labels;
    }

    @Override
    public void setLabels(int idx, INDArray labels) {
        this.labels[idx] = labels;
    }

    @Override
    public boolean hasMaskArrays() {
        if( featuresMaskArrays == null && labelsMaskArrays == null ) return false;
        if(featuresMaskArrays != null){
            for( INDArray i : featuresMaskArrays ){
                if(i != null) return true;
            }
        }
        if(labelsMaskArrays != null){
            for( INDArray i : labelsMaskArrays ){
                if(i != null) return true;
            }
        }
        return false;
    }

    @Override
    public INDArray[] getFeaturesMaskArrays() {
        return featuresMaskArrays;
    }

    @Override
    public INDArray getFeaturesMaskArray(int index) {
        return (featuresMaskArrays != null ? featuresMaskArrays[index] : null);
    }

    @Override
    public void setFeaturesMaskArrays(INDArray[] maskArrays) {
        this.featuresMaskArrays = maskArrays;
    }

    @Override
    public void setFeaturesMaskArray(int idx, INDArray maskArray) {
        this.featuresMaskArrays[idx] = maskArray;
    }

    @Override
    public INDArray[] getLabelsMaskArrays() {
        return labelsMaskArrays;
    }

    @Override
    public INDArray getLabelsMaskArray(int index) {
        return (labelsMaskArrays != null ? labelsMaskArrays[index] : null);
    }

    @Override
    public void setLabelsMaskArray(INDArray[] labelsMaskArrays) {
        this.labelsMaskArrays = labelsMaskArrays;
    }

    @Override
    public void setLabelsMaskArray(int idx, INDArray labelsMaskArray) {
        this.labelsMaskArrays[idx] = labelsMaskArray;
    }


    /** Merge a collection of MultiDataSet objects into a single MultiDataSet.
     * Merging is done by concatenating along dimension 0 (example number in batch)
     * Merging operation may introduce mask arrays (when necessary) for time series data that has different lengths;
     * if mask arrays already exist, these will be merged also.
     *
     * @param toMerge Collection of MultiDataSet objects to merge
     * @return a single MultiDataSet object, containing the arrays of
     */
    public static MultiDataSet merge(Collection<MultiDataSet> toMerge){
        if(toMerge.size() == 1) return toMerge.iterator().next();

        

        throw new UnsupportedOperationException("Not yet implemented");
    }
}
