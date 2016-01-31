package org.nd4j.linalg.dataset;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

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
    public static MultiDataSet merge(Collection<? extends org.nd4j.linalg.dataset.api.MultiDataSet> toMerge){
        if(toMerge.size() == 1){
            org.nd4j.linalg.dataset.api.MultiDataSet mds = toMerge.iterator().next();
            if(mds instanceof MultiDataSet) return (MultiDataSet) mds;
            else return new MultiDataSet(mds.getFeatures(),mds.getLabels(),mds.getFeaturesMaskArrays(),mds.getLabelsMaskArrays());
        }

        List<org.nd4j.linalg.dataset.api.MultiDataSet> list;
        if(toMerge instanceof List) list = (List<org.nd4j.linalg.dataset.api.MultiDataSet>)toMerge;
        else list = new ArrayList<>(toMerge);

        int nInArrays = list.get(0).numFeatureArrays();
        int nOutArrays = list.get(0).numLabelsArrays();

        INDArray[][] features = new INDArray[list.size()][0];
        INDArray[][] labels = new INDArray[list.size()][0];
        INDArray[][] featuresMasks = new INDArray[list.size()][0];
        INDArray[][] labelsMasks = new INDArray[list.size()][0];

        int i=0;
        for( org.nd4j.linalg.dataset.api.MultiDataSet mds : list ){
            features[i] = mds.getFeatures();
            labels[i] = mds.getLabels();
            featuresMasks[i] = mds.getFeaturesMaskArrays();
            labelsMasks[i] = mds.getLabelsMaskArrays();

            if(features[i] == null || features[i].length != nInArrays){
                throw new IllegalStateException("Cannot merge MultiDataSets with different number of input arrays: toMerge[0] has "
                        + nInArrays + " input arrays; toMerge[" + i + "] has " + (features[i] != null ? features[i].length : null) + " arrays");
            }
            if(labels[i] == null || labels[i].length != nOutArrays){
                throw new IllegalStateException("Cannot merge MultiDataSets with different number of output arrays: toMerge[0] has "
                        + nOutArrays + " output arrays; toMerge[" + i + "] has " + (labels[i] != null ? labels[i].length : null) + " arrays");
            }

            i++;
        }

        //Now, merge:
        INDArray[] mergedFeatures = new INDArray[nInArrays];
        INDArray[] mergedLabels = new INDArray[nOutArrays];
        INDArray[] mergedFeaturesMasks = new INDArray[nInArrays];
        INDArray[] mergedLabelsMasks = new INDArray[nOutArrays];

        boolean needFeaturesMasks = false;
        for( i=0; i<nInArrays; i++ ){
            Pair<INDArray,INDArray> pair = merge(features,featuresMasks,i);
            mergedFeatures[i] = pair.getFirst();
            mergedFeaturesMasks[i] = pair.getSecond();
            if(mergedFeaturesMasks[i] != null) needFeaturesMasks = true;
        }
        if(!needFeaturesMasks) mergedFeaturesMasks = null;

        boolean needLabelsMasks = false;
        for( i=0; i<nOutArrays; i++ ){
            Pair<INDArray,INDArray> pair = merge(labels,labelsMasks,i);
            mergedLabels[i] = pair.getFirst();
            mergedLabelsMasks[i] = pair.getSecond();
            if(mergedLabelsMasks[i] != null) needLabelsMasks = true;
        }
        if(!needLabelsMasks) mergedLabelsMasks = null;

        return new MultiDataSet(mergedFeatures,mergedLabels,mergedFeaturesMasks,mergedLabelsMasks);
    }

    private static Pair<INDArray,INDArray> merge(INDArray[][] arrays, INDArray[][] masks, int column){
        int rank = arrays[column][0].rank();
        if(rank == 2){
            return new Pair<>(merge2d(arrays,column),null);
        } else if(rank == 3) {
            return mergeTimeSeries(arrays,masks,column);
        } else if(rank == 4){
            return new Pair<>(merge4d(arrays,column),null);
        } else {
            throw new UnsupportedOperationException("Cannot merge arrays with rank 5 or more (input/output number: " + column + ")");
        }
    }

    private static INDArray merge2d(INDArray[][] arrays, int inOutIdx){
        //Merge 2d data. Mask arrays don't really make sense for 2d, hence are not used here
        int nExamples = 0;
        int cols = arrays[0][inOutIdx].columns();
        for( int i=0; i<arrays.length; i++ ){
            nExamples += arrays[i][inOutIdx].rows();
            if(arrays[i][inOutIdx].columns() != cols){
                throw new IllegalStateException("Cannot merge 2d arrays with different numbers of columns (firstNCols=" + cols
                        + ", ithNCols="+ arrays[i][inOutIdx].columns() + ")");
            }
        }
        INDArray out = Nd4j.create(nExamples,cols);

        int rowsSoFar = 0;
        for( int i=0; i<arrays.length; i++ ){
            int thisRows = arrays[i][inOutIdx].rows();
            out.put(new INDArrayIndex[]{NDArrayIndex.interval(rowsSoFar, rowsSoFar + thisRows),NDArrayIndex.all()},arrays[i][inOutIdx]);
            rowsSoFar += thisRows;
        }
        return out;
    }

    private static Pair<INDArray,INDArray> mergeTimeSeries(INDArray[][] arrays, INDArray[][] masks, int inOutIdx){
        //Merge time series data, and handle masking etc for different length arrays

        //Complications with time series:
        //(a) They may have different lengths (if so: need input + output masking arrays)
        //(b) Even if they are all the same length, they may have masking arrays (if so: merge the masking arrays too)

        int firstLength = arrays[0][inOutIdx].size(2);
        int size = arrays[0][inOutIdx].size(1);
        int maxLength = firstLength;

        boolean hasMask = false;
        boolean lengthsDiffer = false;
        int totalExamples = 0;
        for(int i=0; i<arrays.length; i++ ){
            totalExamples += arrays[i][inOutIdx].size(0);
            int thisLength = arrays[i][inOutIdx].size(2);
            maxLength = Math.max(maxLength,thisLength);
            if( thisLength != firstLength ) lengthsDiffer = true;
            if( masks != null && masks[i] != null && masks[i][inOutIdx] != null ) hasMask = true;

            if(arrays[i][inOutIdx].size(1) != size){
                throw new IllegalStateException("Cannot merge time series with different size for dimension 1 (first shape: "
                    + Arrays.toString(arrays[0][inOutIdx].shape()) + ", " + i + "th shape: " + Arrays.toString(arrays[i][inOutIdx].shape()));
            }
        }

        boolean needMask = hasMask || lengthsDiffer;
        INDArray arr = Nd4j.create(totalExamples,size,maxLength);
        INDArray mask = (needMask ? Nd4j.ones(totalExamples, maxLength) : null);

        //Now, merge the time series (and if necessary, mask arrays):
        int examplesSoFar = 0;
        if(!lengthsDiffer && !needMask){
            //Simplest case: same length, no mask arrays
            for( int i=0; i<arrays.length; i++ ){
                int thisNExamples = arrays[i][inOutIdx].size(0);
                arr.put(new INDArrayIndex[]{NDArrayIndex.interval(examplesSoFar,examplesSoFar+thisNExamples),NDArrayIndex.all(),NDArrayIndex.all()},
                        arrays[i][inOutIdx]);
                examplesSoFar += thisNExamples;
            }
            return new Pair<>(arr,null);
        } else {
            //Either different length, or have mask arrays (or, both)
            for( int i=0; i<arrays.length; i++ ){
                INDArray a = arrays[i][inOutIdx];
                int thisNExamples = a.size(0);
                int thisLength = a.size(2);
                arr.put(new INDArrayIndex[]{NDArrayIndex.interval(examplesSoFar,examplesSoFar+thisNExamples),NDArrayIndex.all(),NDArrayIndex.interval(0,thisLength)},a);

                if(masks != null && masks[i] != null && masks[i][inOutIdx] != null){
                    INDArray origMask = masks[i][inOutIdx];
                    int maskLength = origMask.size(1);
                    mask.put(new INDArrayIndex[]{NDArrayIndex.interval(examplesSoFar,examplesSoFar+thisNExamples), NDArrayIndex.interval(0,maskLength)},origMask);
                    if(maskLength < maxLength){
                        //Set end mask array to zero...
                        mask.put(new INDArrayIndex[]{NDArrayIndex.interval(examplesSoFar,examplesSoFar+thisNExamples), NDArrayIndex.interval(maskLength,maxLength)},
                                Nd4j.zeros(thisNExamples,maxLength-maskLength));
                    }
                } else {
                    if(thisLength<maxLength){
                        //Mask the end
                        mask.put(new INDArrayIndex[]{NDArrayIndex.interval(examplesSoFar,examplesSoFar+thisNExamples), NDArrayIndex.interval(thisLength,maxLength)},
                                Nd4j.zeros(thisNExamples,maxLength-thisLength));
                    }
                }

                examplesSoFar += thisNExamples;
            }
        }

        return new Pair<>(arr,mask);
    }

    private static INDArray merge4d(INDArray[][] arrays, int inOutIdx){
        //4d -> images. Mask arrays for images: not really used

        int nExamples = 0;
        int[] shape = arrays[0][inOutIdx].shape();
        for( int i=0; i<arrays.length; i++ ){
            nExamples += arrays[i][inOutIdx].size(0);
            int[] thisShape = arrays[i][inOutIdx].shape();
            if(thisShape.length != 4){
                throw new IllegalStateException("Cannot merge 4d arrays with non 4d arrays");
            }
            for( int j=1; j<4; j++ ){
                if(thisShape[j] != shape[j]) throw new IllegalStateException("Cannot merge 4d arrays with different shape (other than # examples): "
                        + " data[0][" + inOutIdx + "].shape = " + Arrays.toString(shape) + ", data[" + i + "][" + inOutIdx + "].shape = "
                        + Arrays.toString(thisShape));
            }
        }
        INDArray out = Nd4j.create(nExamples,shape[1],shape[2],shape[3]);

        int rowsSoFar = 0;
        for( int i=0; i<arrays.length; i++ ){
            int thisRows = arrays[i][inOutIdx].size(0);
            out.put(new INDArrayIndex[]{NDArrayIndex.interval(rowsSoFar, rowsSoFar + thisRows), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()},
                    arrays[i][inOutIdx]);
            rowsSoFar += thisRows;
        }
        return out;
    }


}
