package org.nd4j.linalg.dataset;

import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.util.*;

/**Implementation of {@link org.nd4j.linalg.dataset.api.MultiDataSet}
 * @author Alex Black
 */
public class MultiDataSet implements org.nd4j.linalg.dataset.api.MultiDataSet {
    private static final ThreadLocal<INDArray> EMPTY_MASK_ARRAY_PLACEHOLDER = new ThreadLocal<>();

    private INDArray[] features;
    private INDArray[] labels;
    private INDArray[] featuresMaskArrays;
    private INDArray[] labelsMaskArrays;

    private List<Serializable> exampleMetaData;

    /** Create a new (empty) MultiDataSet object (all fields are null) */
    public MultiDataSet() {

    }

    /** MultiDataSet constructor with single features/labels input, no mask arrays */
    public MultiDataSet(INDArray features, INDArray labels) {
        this(features, labels, null, null);
    }

    /** MultiDataSet constructor with single features/labels input, single mask arrays */
    public MultiDataSet(INDArray features, INDArray labels, INDArray featuresMask, INDArray labelsMask) {
        this((features != null ? new INDArray[] {features} : null), (labels != null ? new INDArray[] {labels} : null),
                        (featuresMask != null ? new INDArray[] {featuresMask} : null),
                        (labelsMask != null ? new INDArray[] {labelsMask} : null));
    }

    /** MultiDataSet constructor with no mask arrays */
    public MultiDataSet(INDArray[] features, INDArray[] labels) {
        this(features, labels, null, null);
    }

    /**
     *
     * @param features The features (inputs) to the algorithm/neural network
     * @param labels The labels (outputs) to the algorithm/neural network
     * @param featuresMaskArrays The mask arrays for the features. May be null. Typically used with variable-length time series models, etc
     * @param labelsMaskArrays The mask arrays for the labels. May be null. Typically used with variable-length time series models, etc
     */
    public MultiDataSet(INDArray[] features, INDArray[] labels, INDArray[] featuresMaskArrays,
                    INDArray[] labelsMaskArrays) {
        if (features != null && featuresMaskArrays != null && features.length != featuresMaskArrays.length) {
            throw new IllegalArgumentException("Invalid features / features mask arrays combination: "
                            + "features and features mask arrays must not be different lengths");
        }
        if (labels != null && labelsMaskArrays != null && labels.length != labelsMaskArrays.length) {
            throw new IllegalArgumentException("Invalid labels / labels mask arrays combination: "
                            + "labels and labels mask arrays must not be different lengths");
        }

        this.features = features;
        this.labels = labels;
        this.featuresMaskArrays = featuresMaskArrays;
        this.labelsMaskArrays = labelsMaskArrays;

        Nd4j.getExecutioner().commit();

    }

    @Override
    public List<Serializable> getExampleMetaData() {
        return exampleMetaData;
    }

    @Override
    public <T extends Serializable> List<T> getExampleMetaData(Class<T> metaDataType) {
        return (List<T>) exampleMetaData;
    }

    @Override
    public void setExampleMetaData(List<? extends Serializable> exampleMetaData) {
        this.exampleMetaData = (List<Serializable>) exampleMetaData;
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
        if (featuresMaskArrays == null && labelsMaskArrays == null)
            return false;
        if (featuresMaskArrays != null) {
            for (INDArray i : featuresMaskArrays) {
                if (i != null)
                    return true;
            }
        }
        if (labelsMaskArrays != null) {
            for (INDArray i : labelsMaskArrays) {
                if (i != null)
                    return true;
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

    @Override
    public void save(OutputStream to) throws IOException {
        int numFArr = (features == null ? 0 : features.length);
        int numLArr = (labels == null ? 0 : labels.length);
        int numFMArr = (featuresMaskArrays == null ? 0 : featuresMaskArrays.length);
        int numLMArr = (labelsMaskArrays == null ? 0 : labelsMaskArrays.length);

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(to))) {
            dos.writeInt(numFArr);
            dos.writeInt(numLArr);
            dos.writeInt(numFMArr);
            dos.writeInt(numLMArr);

            saveINDArrays(features, dos, false);
            saveINDArrays(labels, dos, false);
            saveINDArrays(featuresMaskArrays, dos, true);
            saveINDArrays(labelsMaskArrays, dos, true);
        }
    }

    private void saveINDArrays(INDArray[] arrays, DataOutputStream dos, boolean isMask) throws IOException {
        if (arrays != null && arrays.length > 0) {
            for (INDArray fm : arrays) {
                if (isMask && fm == null) {
                    INDArray temp = EMPTY_MASK_ARRAY_PLACEHOLDER.get();
                    if(temp == null){
                        EMPTY_MASK_ARRAY_PLACEHOLDER.set(Nd4j.create(new float[] {-1}));
                        temp = EMPTY_MASK_ARRAY_PLACEHOLDER.get();
                    }
                    fm = temp;
                }
                Nd4j.write(fm, dos);
            }
        }
    }

    @Override
    public void save(File to) throws IOException {
        save(new FileOutputStream(to));
    }

    @Override
    public void load(InputStream from) throws IOException {
        try (DataInputStream dis = new DataInputStream(from)) {
            int numFArr = dis.readInt();
            int numLArr = dis.readInt();
            int numFMArr = dis.readInt();
            int numLMArr = dis.readInt();

            features = loadINDArrays(numFArr, dis, false);
            labels = loadINDArrays(numLArr, dis, false);
            featuresMaskArrays = loadINDArrays(numFMArr, dis, true);
            labelsMaskArrays = loadINDArrays(numLMArr, dis, true);
        }
    }

    private INDArray[] loadINDArrays(int numArrays, DataInputStream dis, boolean isMask) throws IOException {
        INDArray[] result = null;
        if (numArrays > 0) {
            result = new INDArray[numArrays];
            for (int i = 0; i < numArrays; i++) {
                INDArray arr = Nd4j.read(dis);
                result[i] = isMask && arr.equals(EMPTY_MASK_ARRAY_PLACEHOLDER.get()) ? null : arr;
            }
        }
        return result;
    }

    @Override
    public void load(File from) throws IOException {
        load(new FileInputStream(from));
    }

    @Override
    public List<org.nd4j.linalg.dataset.api.MultiDataSet> asList() {
        long nExamples = features[0].size(0);

        List<org.nd4j.linalg.dataset.api.MultiDataSet> list = new ArrayList<>();

        for (int i = 0; i < nExamples; i++) {
            INDArray[] thisFeatures = new INDArray[features.length];
            INDArray[] thisLabels = new INDArray[labels.length];
            INDArray[] thisFeaturesMaskArray =
                            (featuresMaskArrays != null ? new INDArray[featuresMaskArrays.length] : null);
            INDArray[] thisLabelsMaskArray = (labelsMaskArrays != null ? new INDArray[labelsMaskArrays.length] : null);

            for (int j = 0; j < features.length; j++) {
                thisFeatures[j] = getSubsetForExample(features[j], i);
            }
            for (int j = 0; j < labels.length; j++) {
                thisLabels[j] = getSubsetForExample(labels[j], i);
            }
            if (thisFeaturesMaskArray != null) {
                for (int j = 0; j < thisFeaturesMaskArray.length; j++) {
                    if (featuresMaskArrays[j] == null)
                        continue;
                    thisFeaturesMaskArray[j] = getSubsetForExample(featuresMaskArrays[j], i);
                }
            }
            if (thisLabelsMaskArray != null) {
                for (int j = 0; j < thisLabelsMaskArray.length; j++) {
                    if (labelsMaskArrays[j] == null)
                        continue;
                    thisLabelsMaskArray[j] = getSubsetForExample(labelsMaskArrays[j], i);
                }
            }

            list.add(new MultiDataSet(thisFeatures, thisLabels, thisFeaturesMaskArray, thisLabelsMaskArray));
        }

        return list;
    }


    private static INDArray getSubsetForExample(INDArray array, int idx) {
        //Note the interval use here: normally .point(idx) would be used, but this collapses the point dimension
        // when used on arrays with rank of 3 or greater
        //So (point,all,all) on a 3d input returns a 2d output. Whereas, we want a 3d [1,x,y] output here
        switch (array.rank()) {
            case 2:
                return array.get(NDArrayIndex.point(idx), NDArrayIndex.all());
            case 3:
                return array.get(NDArrayIndex.interval(idx, idx, true), NDArrayIndex.all(), NDArrayIndex.all());
            case 4:
                return array.get(NDArrayIndex.interval(idx, idx, true), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all());
            default:
                throw new IllegalStateException("Cannot get subset for rank " + array.rank() + " array");
        }
    }

    /**
     * Clone the dataset
     *
     * @return a clone of the dataset
     */
    @Override
    public MultiDataSet copy() {
        MultiDataSet ret = new MultiDataSet(copy(getFeatures()), copy(getLabels()));
        if (labelsMaskArrays != null) {
            ret.setLabelsMaskArray(copy(labelsMaskArrays));
        }
        if (featuresMaskArrays != null) {
            ret.setFeaturesMaskArrays(copy(featuresMaskArrays));
        }
        return ret;
    }

    private INDArray[] copy(INDArray[] arrays) {
        INDArray[] result = new INDArray[arrays.length];
        for (int i = 0; i < arrays.length; i++) {
            result[i] = arrays[i].dup();
        }
        return result;
    }


    /** Merge a collection of MultiDataSet objects into a single MultiDataSet.
     * Merging is done by concatenating along dimension 0 (example number in batch)
     * Merging operation may introduce mask arrays (when necessary) for time series data that has different lengths;
     * if mask arrays already exist, these will be merged also.
     *
     * @param toMerge Collection of MultiDataSet objects to merge
     * @return a single MultiDataSet object, containing the arrays of
     */
    public static MultiDataSet merge(Collection<? extends org.nd4j.linalg.dataset.api.MultiDataSet> toMerge) {
        if (toMerge.size() == 1) {
            org.nd4j.linalg.dataset.api.MultiDataSet mds = toMerge.iterator().next();
            if (mds instanceof MultiDataSet)
                return (MultiDataSet) mds;
            else
                return new MultiDataSet(mds.getFeatures(), mds.getLabels(), mds.getFeaturesMaskArrays(),
                                mds.getLabelsMaskArrays());
        }

        List<org.nd4j.linalg.dataset.api.MultiDataSet> list;
        if (toMerge instanceof List)
            list = (List<org.nd4j.linalg.dataset.api.MultiDataSet>) toMerge;
        else
            list = new ArrayList<>(toMerge);

        int nonEmpty = 0;
        for(org.nd4j.linalg.dataset.api.MultiDataSet mds : toMerge){
            if(mds.isEmpty()){
                continue;
            }
            nonEmpty++;
        }

        int nInArrays = list.get(0).numFeatureArrays();
        int nOutArrays = list.get(0).numLabelsArrays();

        INDArray[][] features = new INDArray[nonEmpty][0];
        INDArray[][] labels = new INDArray[nonEmpty][0];
        INDArray[][] featuresMasks = new INDArray[nonEmpty][0];
        INDArray[][] labelsMasks = new INDArray[nonEmpty][0];

        int i = 0;
        for (org.nd4j.linalg.dataset.api.MultiDataSet mds : list) {
            if(mds.isEmpty()){
                continue;
            }

            features[i] = mds.getFeatures();
            labels[i] = mds.getLabels();
            featuresMasks[i] = mds.getFeaturesMaskArrays();
            labelsMasks[i] = mds.getLabelsMaskArrays();

            if (features[i] == null || features[i].length != nInArrays) {
                throw new IllegalStateException(
                                "Cannot merge MultiDataSets with different number of input arrays: toMerge[0] has "
                                                + nInArrays + " input arrays; toMerge[" + i + "] has "
                                                + (features[i] != null ? features[i].length : null) + " arrays");
            }
            if (labels[i] == null || labels[i].length != nOutArrays) {
                throw new IllegalStateException(
                                "Cannot merge MultiDataSets with different number of output arrays: toMerge[0] has "
                                                + nOutArrays + " output arrays; toMerge[" + i + "] has "
                                                + (labels[i] != null ? labels[i].length : null) + " arrays");
            }

            i++;
        }

        //Now, merge:
        INDArray[] mergedFeatures = new INDArray[nInArrays];
        INDArray[] mergedLabels = new INDArray[nOutArrays];
        INDArray[] mergedFeaturesMasks = new INDArray[nInArrays];
        INDArray[] mergedLabelsMasks = new INDArray[nOutArrays];

        boolean needFeaturesMasks = false;
        for (i = 0; i < nInArrays; i++) {
            Pair<INDArray, INDArray> pair = DataSetUtil.mergeFeatures(features, featuresMasks, i); //merge(features, featuresMasks, i);
            mergedFeatures[i] = pair.getFirst();
            mergedFeaturesMasks[i] = pair.getSecond();
            if (mergedFeaturesMasks[i] != null)
                needFeaturesMasks = true;
        }
        if (!needFeaturesMasks)
            mergedFeaturesMasks = null;

        boolean needLabelsMasks = false;
        for (i = 0; i < nOutArrays; i++) {
            Pair<INDArray, INDArray> pair = DataSetUtil.mergeLabels(labels, labelsMasks, i);
            mergedLabels[i] = pair.getFirst();
            mergedLabelsMasks[i] = pair.getSecond();
            if (mergedLabelsMasks[i] != null)
                needLabelsMasks = true;
        }
        if (!needLabelsMasks)
            mergedLabelsMasks = null;

        return new MultiDataSet(mergedFeatures, mergedLabels, mergedFeaturesMasks, mergedLabelsMasks);
    }


    @Override
    public String toString() {
        int nfMask = 0;
        int nlMask = 0;
        if(featuresMaskArrays != null){
            for(INDArray i : featuresMaskArrays){
                if(i != null){
                    nfMask++;
                }
            }
        }
        if(labelsMaskArrays != null){
            for(INDArray i : labelsMaskArrays){
                if(i != null){
                    nlMask++;
                }
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("MultiDataSet: ").append(numFeatureArrays()).append(" input arrays, ")
                .append(numLabelsArrays()).append(" label arrays, ")
                .append(nfMask).append(" input masks, ")
                .append(nlMask).append(" label masks");


        for (int i = 0; i < numFeatureArrays(); i++) {
            sb.append("\n=== INPUT ").append(i).append(" ===\n").append(getFeatures(i).toString().replaceAll(";", "\n"));
            if (getFeaturesMaskArray(i) != null) {
                sb.append("\n--- INPUT MASK ---\n")
                        .append(getFeaturesMaskArray(i).toString().replaceAll(";", "\n"));
            }
        }
        for( int i=0; i<numLabelsArrays(); i++){
            sb.append("\n=== LABEL ").append(i).append(" ===\n")
                    .append(getLabels(i).toString().replaceAll(";", "\n"));

            if (getLabelsMaskArray(i) != null) {
                sb.append("\n--- LABEL MASK ---\n")
                        .append(getLabelsMaskArray(i).toString().replaceAll(";", "\n"));
            }
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (!(o instanceof MultiDataSet))
            return false;

        MultiDataSet m = (MultiDataSet) o;

        if (!bothNullOrEqual(features, m.features))
            return false;
        if (!bothNullOrEqual(labels, m.labels))
            return false;
        if (!bothNullOrEqual(featuresMaskArrays, m.featuresMaskArrays))
            return false;
        return bothNullOrEqual(labelsMaskArrays, m.labelsMaskArrays);
    }

    private boolean bothNullOrEqual(INDArray[] first, INDArray[] second) {
        if (first == null && second == null)
            return true;
        if (first == null || second == null)
            return false; //One but not both null
        if (first.length != second.length)
            return false;
        for (int i = 0; i < first.length; i++) {
            if (!Objects.equals(first[i], second[i])) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = 0;
        if (features != null) {
            for (INDArray f : features) {
                result = result * 31 + f.hashCode();
            }
        }
        if (labels != null) {
            for (INDArray l : labels) {
                result = result * 31 + l.hashCode();
            }
        }
        if (featuresMaskArrays != null) {
            for (INDArray fm : featuresMaskArrays) {
                result = result * 31 + fm.hashCode();
            }
        }
        if (labelsMaskArrays != null) {
            for (INDArray lm : labelsMaskArrays) {
                result = result * 31 + lm.hashCode();
            }
        }
        return result;
    }

    /**
     * This method returns memory used by this DataSet
     *
     * @return
     */
    @Override
    public long getMemoryFootprint() {
        long reqMem = 0;

        for (INDArray f : features)
            reqMem += f == null ? 0 : f.lengthLong() * Nd4j.sizeOfDataType();

        if (featuresMaskArrays != null)
            for (INDArray f : featuresMaskArrays)
                reqMem += f == null ? 0 : f.lengthLong() * Nd4j.sizeOfDataType();

        if (labelsMaskArrays != null)
            for (INDArray f : labelsMaskArrays)
                reqMem += f == null ? 0 : f.lengthLong() * Nd4j.sizeOfDataType();

        if (labels != null)
            for (INDArray f : labels)
                reqMem += f == null ? 0 : f.lengthLong() * Nd4j.sizeOfDataType();

        return reqMem;
    }


    @Override
    public void migrate() {
        if (Nd4j.getMemoryManager().getCurrentWorkspace() != null) {
            if (features != null)
                for (int e = 0; e < features.length; e++)
                    features[e] = features[e].migrate();

            if (labels != null)
                for (int e = 0; e < labels.length; e++)
                    labels[e] = labels[e].migrate();

            if (featuresMaskArrays != null)
                for (int e = 0; e < featuresMaskArrays.length; e++)
                    featuresMaskArrays[e] = featuresMaskArrays[e].migrate();

            if (labelsMaskArrays != null)
                for (int e = 0; e < labelsMaskArrays.length; e++)
                    labelsMaskArrays[e] = labelsMaskArrays[e].migrate();
        }
    }

    /**
     * This method migrates this DataSet into current Workspace (if any)
     */
    @Override
    public void detach() {
        if (features != null)
            for (int e = 0; e < features.length; e++)
                features[e] = features[e].detach();

        if (labels != null)
            for (int e = 0; e < labels.length; e++)
                labels[e] = labels[e].detach();

        if (featuresMaskArrays != null)
            for (int e = 0; e < featuresMaskArrays.length; e++)
                featuresMaskArrays[e] = featuresMaskArrays[e].detach();

        if (labelsMaskArrays != null)
            for (int e = 0; e < labelsMaskArrays.length; e++)
                labelsMaskArrays[e] = labelsMaskArrays[e].detach();
    }

    @Override
    public boolean isEmpty() {
        return nullOrEmpty(features) && nullOrEmpty(labels) && nullOrEmpty(featuresMaskArrays) && nullOrEmpty(labelsMaskArrays);
    }

    @Override
    public void shuffle() {
        List<org.nd4j.linalg.dataset.api.MultiDataSet> split = asList();
        Collections.shuffle(split);
        MultiDataSet mds = merge(split);
        this.features = mds.features;
        this.labels = mds.labels;
        this.featuresMaskArrays = mds.featuresMaskArrays;
        this.labelsMaskArrays = mds.labelsMaskArrays;
        this.exampleMetaData = mds.exampleMetaData;
    }

    private static boolean nullOrEmpty(INDArray[] arr){
        if(arr == null){
            return true;
        }
        for(INDArray i : arr){
            if(i != null){
                return false;
            }
        }
        return true;
    }
}
