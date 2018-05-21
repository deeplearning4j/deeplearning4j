/*-
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
 *
 */

package org.nd4j.linalg.dataset;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.util.MathUtils;

import java.io.*;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;


/**
 * A data transform (example/outcome pairs)
 * The outcomes are specifically for neural network encoding such that
 * any labels that are considered true are 1s. The rest are zeros.
 *
 * @author Adam Gibson
 */
@Slf4j
public class DataSet implements org.nd4j.linalg.dataset.api.DataSet {

    private static final long serialVersionUID = 1935520764586513365L;

    private static final byte BITMASK_FEATURES_PRESENT = 1;
    private static final byte BITMASK_LABELS_PRESENT = 1 << 1;
    private static final byte BITMASK_LABELS_SAME_AS_FEATURES = 1 << 2;
    private static final byte BITMASK_FEATURE_MASK_PRESENT = 1 << 3;
    private static final byte BITMASK_LABELS_MASK_PRESENT = 1 << 4;

    private List<String> columnNames = new ArrayList<>();
    private List<String> labelNames = new ArrayList<>();
    private INDArray features, labels;
    private INDArray featuresMask;
    private INDArray labelsMask;

    private List<Serializable> exampleMetaData;

    private transient boolean preProcessed = false;

    public DataSet() {
        this(null, null);
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


    /**
     * Creates a dataset with the specified input matrix and labels
     *
     * @param first  the feature matrix
     * @param second the labels (these should be binarized label matrices such that the specified label
     *               has a value of 1 in the desired column with the label)
     */
    public DataSet(INDArray first, INDArray second) {
        this(first, second, null, null);
    }

    /**Create a dataset with the specified input INDArray and labels (output) INDArray, plus (optionally) mask arrays
     * for the features and labels
     * @param features Features (input)
     * @param labels Labels (output)
     * @param featuresMask Mask array for features, may be null
     * @param labelsMask Mask array for labels, may be null
     */
    public DataSet(INDArray features, INDArray labels, INDArray featuresMask, INDArray labelsMask) {
        this.features = features;
        this.labels = labels;
        this.featuresMask = featuresMask;
        this.labelsMask = labelsMask;

        // we want this dataset to be fully committed to device
        Nd4j.getExecutioner().commit();
    }

    public boolean isPreProcessed() {
        return preProcessed;
    }

    public void markAsPreProcessed() {
        this.preProcessed = true;
    }

    /**
     * Returns a single dataset (all fields are null)
     *
     * @return an empty dataset (all fields are null)
     */
    public static DataSet empty() {
        return new DataSet(null, null);
    }

    /**
     * Merge the list of datasets in to one list.
     * All the rows are merged in to one dataset
     *
     * @param data the data to merge
     * @return a single dataset
     */
    public static DataSet merge(List<DataSet> data) {
        if (data.isEmpty())
            throw new IllegalArgumentException("Unable to merge empty dataset");

        int nonEmpty = 0;
        boolean anyFeaturesPreset = false;
        boolean anyLabelsPreset = false;
        boolean first = true;
        for(DataSet ds : data){
            if(ds.isEmpty()){
                continue;
            }
            nonEmpty++;

            if(anyFeaturesPreset && ds.getFeatures() == null || (!first && !anyFeaturesPreset && ds.getFeatures() != null)){
                throw new IllegalStateException("Cannot merge features: encountered null features in one or more DataSets");
            }
            if(anyLabelsPreset && ds.getLabels() == null || (!first && !anyLabelsPreset && ds.getLabels() != null)){
                throw new IllegalStateException("Cannot merge labels: enountered null labels in one or more DataSets");
            }

            anyFeaturesPreset |= ds.getFeatures() != null;
            anyLabelsPreset |= ds.getLabels() != null;
            first = false;
        }

        INDArray[] featuresToMerge = new INDArray[nonEmpty];
        INDArray[] labelsToMerge = new INDArray[nonEmpty];
        INDArray[] featuresMasksToMerge = null;
        INDArray[] labelsMasksToMerge = null;
        int count = 0;
        for (DataSet ds : data) {
            if(ds.isEmpty())
                continue;
            featuresToMerge[count] = ds.getFeatureMatrix();
            labelsToMerge[count] = ds.getLabels();

            if (ds.getFeaturesMaskArray() != null) {
                if (featuresMasksToMerge == null) {
                    featuresMasksToMerge = new INDArray[data.size()];
                }
                featuresMasksToMerge[count] = ds.getFeaturesMaskArray();
            }
            if (ds.getLabelsMaskArray() != null) {
                if (labelsMasksToMerge == null) {
                    labelsMasksToMerge = new INDArray[data.size()];
                }
                labelsMasksToMerge[count] = ds.getLabelsMaskArray();
            }

            count++;
        }

        INDArray featuresOut;
        INDArray labelsOut;
        INDArray featuresMaskOut;
        INDArray labelsMaskOut;

        Pair<INDArray, INDArray> fp = DataSetUtil.mergeFeatures(featuresToMerge, featuresMasksToMerge);
        featuresOut = fp.getFirst();
        featuresMaskOut = fp.getSecond();

        Pair<INDArray, INDArray> lp = DataSetUtil.mergeLabels(labelsToMerge, labelsMasksToMerge);
        labelsOut = lp.getFirst();
        labelsMaskOut = lp.getSecond();

        DataSet dataset = new DataSet(featuresOut, labelsOut, featuresMaskOut, labelsMaskOut);

        List<Serializable> meta = null;
        for (DataSet ds : data) {
            if (ds.getExampleMetaData() == null || ds.getExampleMetaData().size() != ds.numExamples()) {
                meta = null;
                break;
            }
            if (meta == null)
                meta = new ArrayList<>();
            meta.addAll(ds.getExampleMetaData());
        }
        if (meta != null) {
            dataset.setExampleMetaData(meta);
        }

        return dataset;
    }

    @Override
    public org.nd4j.linalg.dataset.api.DataSet getRange(int from, int to) {
        if (hasMaskArrays()) {
            INDArray featureMaskHere = featuresMask != null ? featuresMask.get(interval(from, to)) : null;
            INDArray labelMaskHere = labelsMask != null ? labelsMask.get(interval(from, to)) : null;
            return new DataSet(features.get(interval(from, to)), labels.get(interval(from, to)), featureMaskHere,
                            labelMaskHere);
        }
        return new DataSet(features.get(interval(from, to)), labels.get(interval(from, to)));
    }


    @Override
    public void load(InputStream from) {
        try {

            DataInputStream dis = from instanceof BufferedInputStream ? new DataInputStream(from)
                            : new DataInputStream(new BufferedInputStream(from));

            byte included = dis.readByte();
            boolean hasFeatures = (included & BITMASK_FEATURES_PRESENT) != 0;
            boolean hasLabels = (included & BITMASK_LABELS_PRESENT) != 0;
            boolean hasLabelsSameAsFeatures = (included & BITMASK_LABELS_SAME_AS_FEATURES) != 0;
            boolean hasFeaturesMask = (included & BITMASK_FEATURE_MASK_PRESENT) != 0;
            boolean hasLabelsMask = (included & BITMASK_LABELS_MASK_PRESENT) != 0;

            features = (hasFeatures ? Nd4j.read(dis) : null);
            if (hasLabels) {
                labels = Nd4j.read(dis);
            } else if (hasLabelsSameAsFeatures) {
                labels = features;
            } else {
                labels = null;
            }

            featuresMask = (hasFeaturesMask ? Nd4j.read(dis) : null);
            labelsMask = (hasLabelsMask ? Nd4j.read(dis) : null);

            dis.close();
        } catch (Exception e) {
            throw new RuntimeException("Error loading DataSet",e);
        }
    }

    @Override
    public void load(File from) {
        try (FileInputStream fis = new FileInputStream(from);
                        BufferedInputStream bis = new BufferedInputStream(fis, 1024 * 1024)) {
            load(bis);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public void save(OutputStream to) {

        byte included = 0;
        if (features != null)
            included |= BITMASK_FEATURES_PRESENT;
        if (labels != null) {
            if (labels == features) {
                //Same object. Don't serialize the same data twice!
                included |= BITMASK_LABELS_SAME_AS_FEATURES;
            } else {
                included |= BITMASK_LABELS_PRESENT;
            }
        }
        if (featuresMask != null)
            included |= BITMASK_FEATURE_MASK_PRESENT;
        if (labelsMask != null)
            included |= BITMASK_LABELS_MASK_PRESENT;


        try {
            BufferedOutputStream bos = new BufferedOutputStream(to);
            DataOutputStream dos = new DataOutputStream(bos);
            dos.writeByte(included);

            if (features != null)
                Nd4j.write(features, dos);
            if (labels != null && labels != features)
                Nd4j.write(labels, dos);
            if (featuresMask != null)
                Nd4j.write(featuresMask, dos);
            if (labelsMask != null)
                Nd4j.write(labelsMask, dos);

            dos.flush();
            dos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void save(File to) {
        try (FileOutputStream fos = new FileOutputStream(to, false);
                        BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            save(bos);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public DataSetIterator iterateWithMiniBatches() {
        return null;
    }

    @Override
    public String id() {
        return "";
    }

    @Override
    public INDArray getFeatures() {
        return features;
    }

    @Override
    public void setFeatures(INDArray features) {
        this.features = features;
    }

    @Override
    public Map<Integer, Double> labelCounts() {
        Map<Integer, Double> ret = new HashMap<>();
        if (labels == null)
            return ret;
        long nTensors = labels.tensorssAlongDimension(1);
        for (int i = 0; i < nTensors; i++) {
            INDArray row = labels.tensorAlongDimension(i, 1);
            INDArray javaRow = labels.javaTensorAlongDimension(i, 1);
            int maxIdx = Nd4j.getBlasWrapper().iamax(row);
            int maxIdxJava = Nd4j.getBlasWrapper().iamax(javaRow);
            if (maxIdx < 0)
                throw new IllegalStateException("Please check the iamax implementation for "
                                + Nd4j.getBlasWrapper().getClass().getName());
            if (ret.get(maxIdx) == null)
                ret.put(maxIdx, 1.0);
            else
                ret.put(maxIdx, ret.get(maxIdx) + 1.0);
        }
        return ret;
    }

    @Override
    public void apply(Condition condition, Function<Number, Number> function) {
        BooleanIndexing.applyWhere(getFeatureMatrix(), condition, function);
    }

    /**
     * Clone the dataset
     *
     * @return a clone of the dataset
     */
    @Override
    public DataSet copy() {
        DataSet ret = new DataSet(getFeatures().dup(), getLabels().dup());
        if (getLabelsMaskArray() != null)
            ret.setLabelsMaskArray(getLabelsMaskArray().dup());
        if (getFeaturesMaskArray() != null)
            ret.setFeaturesMaskArray(getFeaturesMaskArray().dup());
        ret.setColumnNames(getColumnNames());
        ret.setLabelNames(getLabelNames());
        return ret;
    }

    /**
     * Reshapes the input in to the given rows and columns
     *
     * @param rows the row size
     * @param cols the column size
     * @return a copy of this data op with the input resized
     */
    @Override
    public DataSet reshape(int rows, int cols) {
        DataSet ret = new DataSet(getFeatures().reshape(new long[] {rows, cols}), getLabels());
        return ret;
    }


    @Override
    public void multiplyBy(double num) {
        getFeatures().muli(Nd4j.scalar(num));
    }

    @Override
    public void divideBy(int num) {
        getFeatures().divi(Nd4j.scalar(num));
    }

    @Override
    public void shuffle() {
        long seed = System.currentTimeMillis();
        shuffle(seed);
    }

    /**
     * Shuffles the dataset in place, given a seed for a random number generator. For reproducibility
     * This will modify the dataset in place!!
     *
     * @param seed Seed to use for the random Number Generator
     */
    public void shuffle(long seed) {
        // just skip shuffle if there's only 1 example
        if (numExamples() < 2)
            return;

        //note here we use the same seed with different random objects guaranteeing same order

        List<INDArray> arrays = new ArrayList<>();
        List<int[]> dimensions = new ArrayList<>();

        arrays.add(getFeatures());
        dimensions.add(ArrayUtil.range(1, getFeatures().rank()));

        arrays.add(getLabels());
        dimensions.add(ArrayUtil.range(1, getLabels().rank()));

        if (featuresMask != null) {
            arrays.add(getFeaturesMaskArray());
            dimensions.add(ArrayUtil.range(1, getFeaturesMaskArray().rank()));
        }

        if (labelsMask != null) {
            arrays.add(getLabelsMaskArray());
            dimensions.add(ArrayUtil.range(1, getLabelsMaskArray().rank()));
        }

        Nd4j.shuffle(arrays, new Random(seed), dimensions);

        //As per CpuNDArrayFactory.shuffle(List<INDArray> arrays, Random rnd, List<int[]> dimensions) and libnd4j transforms.h shuffleKernelGeneric
        if (exampleMetaData != null) {
            int[] map = ArrayUtil.buildInterleavedVector(new Random(seed), numExamples());
            ArrayUtil.shuffleWithMap(exampleMetaData, map);
        }
    }


    /**
     * Squeezes input data to a max and a min
     *
     * @param min the min value to occur in the dataset
     * @param max the max value to ccur in the dataset
     */
    @Override
    public void squishToRange(double min, double max) {
        for (int i = 0; i < getFeatures().length(); i++) {
            double curr = (double) getFeatures().getScalar(i).element();
            if (curr < min)
                getFeatures().put(i, Nd4j.scalar(min));
            else if (curr > max)
                getFeatures().put(i, Nd4j.scalar(max));
        }
    }

    @Override
    public void scaleMinAndMax(double min, double max) {
        FeatureUtil.scaleMinMax(min, max, getFeatureMatrix());
    }

    /**
     * Divides the input data transform
     * by the max number in each row
     */
    @Override
    public void scale() {
        FeatureUtil.scaleByMax(getFeatures());
    }

    /**
     * Adds a feature for each example on to the current feature vector
     *
     * @param toAdd the feature vector to add
     */
    @Override
    public void addFeatureVector(INDArray toAdd) {
        setFeatures(Nd4j.hstack(getFeatureMatrix(), toAdd));
    }


    /**
     * The feature to add, and the example/row number
     *
     * @param feature the feature vector to add
     * @param example the number of the example to append to
     */
    @Override
    public void addFeatureVector(INDArray feature, int example) {
        getFeatures().putRow(example, feature);
    }

    @Override
    public void normalize() {
        //FeatureUtil.normalizeMatrix(getFeatures());
        NormalizerStandardize inClassPreProcessor = new NormalizerStandardize();
        inClassPreProcessor.fit(this);
        inClassPreProcessor.transform(this);
    }


    /**
     * Same as calling binarize(0)
     */
    @Override
    public void binarize() {
        binarize(0);
    }

    /**
     * Binarizes the dataset such that any number greater than cutoff is 1 otherwise zero
     *
     * @param cutoff the cutoff point
     */
    @Override
    public void binarize(double cutoff) {
        INDArray linear = getFeatureMatrix().linearView();
        for (int i = 0; i < getFeatures().length(); i++) {
            double curr = linear.getDouble(i);
            if (curr > cutoff)
                getFeatures().putScalar(i, 1);
            else
                getFeatures().putScalar(i, 0);
        }
    }


    /**
     * @Deprecated
     * Subtract by the column means and divide by the standard deviation
     */
    @Deprecated
    @Override
    public void normalizeZeroMeanZeroUnitVariance() {
        INDArray columnMeans = getFeatures().mean(0);
        INDArray columnStds = getFeatureMatrix().std(0);

        setFeatures(getFeatures().subiRowVector(columnMeans));
        columnStds.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        setFeatures(getFeatures().diviRowVector(columnStds));
    }

    /**
     * The number of inputs in the feature matrix
     *
     * @return
     */
    @Override
    public int numInputs() {
        // FIXME: int cast
        return (int) getFeatures().size(1);
    }

    @Override
    public void validate() {
        if (getFeatures().size(0) != getLabels().size(0))
            throw new IllegalStateException("Invalid dataset");
    }

    @Override
    public int outcome() {
        return Nd4j.getBlasWrapper().iamax(getLabels());
    }

    /**
     * Clears the outcome matrix setting a new number of labels
     *
     * @param labels the number of labels/columns in the outcome matrix
     *               Note that this clears the labels for each example
     */
    @Override
    public void setNewNumberOfLabels(int labels) {
        int examples = numExamples();
        INDArray newOutcomes = Nd4j.create(examples, labels);
        setLabels(newOutcomes);
    }

    /**
     * Sets the outcome of a particular example
     *
     * @param example the example to transform
     * @param label   the label of the outcome
     */
    @Override
    public void setOutcome(int example, int label) {
        if (example > numExamples())
            throw new IllegalArgumentException("No example at " + example);
        if (label > numOutcomes() || label < 0)
            throw new IllegalArgumentException("Illegal label");

        INDArray outcome = FeatureUtil.toOutcomeVector(label, numOutcomes());
        getLabels().putRow(example, outcome);
    }

    /**
     * Gets a copy of example i
     *
     * @param i the example to getFromOrigin
     * @return the example at i (one example)
     */
    @Override
    public DataSet get(int i) {
        if (i > numExamples() || i < 0)
            throw new IllegalArgumentException("invalid example number");
        if (i == 0 && numExamples() == 1)
            return this;
        if (getFeatureMatrix().rank() == 4) {
            //ensure rank is preserved
            INDArray slice = getFeatureMatrix().slice(i);
            return new DataSet(slice.reshape(ArrayUtil.combine(new long[] {1}, slice.shape())), getLabels().slice(i));
        }
        return new DataSet(getFeatures().slice(i), getLabels().slice(i));
    }

    /**
     * Gets a copy of example i
     *
     * @param i the example to getFromOrigin
     * @return the example at i (one example)
     */
    @Override
    public DataSet get(int[] i) {
        return new DataSet(getFeatures().getRows(i), getLabels().getRows(i));
    }

    /**
     * Partitions a dataset in to mini batches where
     * each dataset in each list is of the specified number of examples
     *
     * @param num the number to split by
     * @return the partitioned datasets
     */
    @Override
    public List<DataSet> batchBy(int num) {
        List<DataSet> batched = Lists.newArrayList();
        for (List<DataSet> splitBatch : Lists.partition(asList(), num)) {
            batched.add(DataSet.merge(splitBatch));
        }
        return batched;
    }

    /**
     * Strips the data transform of all but the passed in labels
     *
     * @param labels strips the data transform of all but the passed in labels
     * @return the dataset with only the specified labels
     */
    @Override
    public DataSet filterBy(int[] labels) {
        List<DataSet> list = asList();
        List<DataSet> newList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        for (int i : labels)
            labelList.add(i);
        for (DataSet d : list) {
            int outcome = d.outcome();
            if (labelList.contains(outcome)) {
                newList.add(d);
            }
        }

        return DataSet.merge(newList);
    }

    /**
     * Strips the dataset down to the specified labels
     * and remaps them
     *
     * @param labels the labels to strip down to
     */
    @Override
    public void filterAndStrip(int[] labels) {
        DataSet filtered = filterBy(labels);
        List<Integer> newLabels = new ArrayList<>();

        //map new labels to index according to passed in labels
        Map<Integer, Integer> labelMap = new HashMap<>();

        for (int i = 0; i < labels.length; i++)
            labelMap.put(labels[i], i);

        //map examples
        for (int i = 0; i < filtered.numExamples(); i++) {
            DataSet example = filtered.get(i);
            int o2 = example.outcome();
            Integer outcome = labelMap.get(o2);
            newLabels.add(outcome);

        }


        INDArray newLabelMatrix = Nd4j.create(filtered.numExamples(), labels.length);

        if (newLabelMatrix.rows() != newLabels.size())
            throw new IllegalStateException("Inconsistent label sizes");

        for (int i = 0; i < newLabelMatrix.rows(); i++) {
            Integer i2 = newLabels.get(i);
            if (i2 == null)
                throw new IllegalStateException("Label not found on row " + i);
            INDArray newRow = FeatureUtil.toOutcomeVector(i2, labels.length);
            newLabelMatrix.putRow(i, newRow);

        }

        setFeatures(filtered.getFeatures());
        setLabels(newLabelMatrix);
    }

    /**
     * Partitions the data transform by the specified number.
     *
     * @param num the number to split by
     * @return the partitioned data transform
     */
    @Override
    public List<DataSet> dataSetBatches(int num) {
        List<List<DataSet>> list = Lists.partition(asList(), num);
        List<DataSet> ret = new ArrayList<>();
        for (List<DataSet> l : list)
            ret.add(DataSet.merge(l));
        return ret;

    }

    /**
     * Sorts the dataset by label:
     * Splits the data transform such that examples are sorted by their labels.
     * A ten label dataset would produce lists with batches like the following:
     * x1   y = 1
     * x2   y = 2
     * ...
     * x10  y = 10
     *
     * @return a list of data sets partitioned by outcomes
     */
    @Override
    public List<DataSet> sortAndBatchByNumLabels() {
        sortByLabel();
        return batchByNumLabels();
    }

    @Override
    public List<DataSet> batchByNumLabels() {
        return batchBy(numOutcomes());
    }

    @Override
    public List<DataSet> asList() {
        List<DataSet> list = new ArrayList<>(numExamples());
        INDArray featuresHere, labelsHere, featureMaskHere, labelMaskHere;
        int rank = getFeatures().rank();
        int labelsRank = getLabels().rank();

        // Preserving the dimension of the dataset - essentially a minibatch size of 1
        for (int i = 0; i < numExamples(); i++) {
            switch (rank) {
                case 2:
                    featuresHere = getFeatures().get(interval(i, i, true), all());
                    featureMaskHere = featuresMask != null ? featuresMask.get(interval(i, i, true), all()) : null;
                    break;
                case 3:
                    featuresHere = getFeatures().get(interval(i, i, true), all(), all());
                    featureMaskHere = featuresMask != null ? featuresMask.get(interval(i, i, true), all()) : null;
                    break;
                case 4:
                    featuresHere = getFeatures().get(interval(i, i, true), all(), all(), all());
                    featureMaskHere = featuresMask != null ? featuresMask.get(interval(i, i, true), all()) : null;
                    break;
                default:
                    throw new IllegalStateException(
                                    "Cannot convert to list: feature set rank must be in range 2 to 4 inclusive. First example labels shape: "
                                                    + Arrays.toString(getFeatures().shape()));
            }
            switch (labelsRank) {
                case 2:
                    labelsHere = getLabels().get(interval(i, i, true), all());
                    labelMaskHere = labelsMask != null ? labelsMask.get(interval(i, i, true), all()) : null;
                    break;
                case 3:
                    labelsHere = getLabels().get(interval(i, i, true), all(), all());
                    labelMaskHere = labelsMask != null ? labelsMask.get(interval(i, i, true), all()) : null;
                    break;
                case 4:
                    labelsHere = getLabels().get(interval(i, i, true), all(), all(), all());
                    labelMaskHere = labelsMask != null ? labelsMask.get(interval(i, i, true), all()) : null;
                    break;
                default:
                    throw new IllegalStateException(
                                    "Cannot convert to list: feature set rank must be in range 2 to 4 inclusive. First example labels shape: "
                                                    + Arrays.toString(getFeatures().shape()));

            }

            DataSet ds = new DataSet(featuresHere, labelsHere, featureMaskHere, labelMaskHere);
            if (exampleMetaData != null && exampleMetaData.size() > i) {
                ds.setExampleMetaData(Collections.singletonList(exampleMetaData.get(i)));
            }
            list.add(ds);
        }
        return list;
    }

    /**
     * Splits a dataset in to test and train randomly.
     * This will modify the dataset in place to shuffle it before splitting into test/train!
     *
     * @param numHoldout the number to hold out for training
     * @param  rng Random Number Generator to use to shuffle the dataset
     * @return the pair of datasets for the train test split
     */
    @Override
    public SplitTestAndTrain splitTestAndTrain(int numHoldout, Random rng) {
        long seed = rng.nextLong();
        this.shuffle(seed);
        return splitTestAndTrain(numHoldout);
    }

    /**
     * Splits a dataset in to test and train
     *
     * @param numHoldout the number to hold out for training
     * @return the pair of datasets for the train test split
     */
    @Override
    public SplitTestAndTrain splitTestAndTrain(int numHoldout) {
        int numExamples = numExamples();
        if (numExamples <= 1)
            throw new IllegalStateException(
                            "Cannot split DataSet with <= 1 rows (data set has " + numExamples + " example)");
        if (numHoldout >= numExamples)
            throw new IllegalArgumentException(
                            "Unable to split on size equal or larger than the number of rows (# numExamples="
                                            + numExamples + ", numHoldout=" + numHoldout + ")");
        DataSet first = new DataSet();
        DataSet second = new DataSet();
        switch (features.rank()) {
            case 2:
                first.setFeatures(features.get(interval(0, numHoldout), all()));
                second.setFeatures(features.get(interval(numHoldout, numExamples), all()));
                break;
            case 3:
                first.setFeatures(features.get(interval(0, numHoldout), all(), all()));
                second.setFeatures(features.get(interval(numHoldout, numExamples), all(), all()));
                break;
            case 4:
                first.setFeatures(features.get(interval(0, numHoldout), all(), all(), all()));
                second.setFeatures(features.get(interval(numHoldout, numExamples), all(), all(), all()));
                break;
            default:
                throw new UnsupportedOperationException("Features rank: " + features.rank());
        }
        switch (labels.rank()) {
            case 2:
                first.setLabels(labels.get(interval(0, numHoldout), all()));
                second.setLabels(labels.get(interval(numHoldout, numExamples), all()));
                break;
            case 3:
                first.setLabels(labels.get(interval(0, numHoldout), all(), all()));
                second.setLabels(labels.get(interval(numHoldout, numExamples), all(), all()));
                break;
            case 4:
                first.setLabels(labels.get(interval(0, numHoldout), all(), all(), all()));
                second.setLabels(labels.get(interval(numHoldout, numExamples), all(), all(), all()));
                break;
            default:
                throw new UnsupportedOperationException("Labels rank: " + features.rank());
        }

        if (featuresMask != null) {
            first.setFeaturesMaskArray(featuresMask.get(interval(0, numHoldout), all()));
            second.setFeaturesMaskArray(featuresMask.get(interval(numHoldout, numExamples), all()));
        }
        if (labelsMask != null) {
            first.setLabelsMaskArray(labelsMask.get(interval(0, numHoldout), all()));
            second.setLabelsMaskArray(labelsMask.get(interval(numHoldout, numExamples), all()));
        }

        if (exampleMetaData != null) {
            List<Serializable> meta1 = new ArrayList<>();
            List<Serializable> meta2 = new ArrayList<>();
            for (int i = 0; i < numHoldout && i < exampleMetaData.size(); i++) {
                meta1.add(exampleMetaData.get(i));
            }
            for (int i = numHoldout; i < numExamples && i < exampleMetaData.size(); i++) {
                meta2.add(exampleMetaData.get(i));
            }
            first.setExampleMetaData(meta1);
            second.setExampleMetaData(meta2);
        }
        return new SplitTestAndTrain(first, second);
    }


    /**
     * Returns the labels for the dataset
     *
     * @return the labels for the dataset
     */
    @Override
    public INDArray getLabels() {
        return labels;
    }

    /**
     * @param idx the index to pullRows the string label value out of the list if it exists
     * @return the label opName
     */
    @Override
    public String getLabelName(int idx) {
        if (!labelNames.isEmpty()) {
            if (idx < labelNames.size())
                return labelNames.get(idx);
            else
                throw new IllegalStateException(
                                "Index requested is longer than the number of labels used for classification.");
        } else
            throw new IllegalStateException(
                            "Label names are not defined on this dataset. Add label names in order to use getLabelName with an id.");

    }

    /**
     * @param idxs list of index to pullRows the string label value out of the list if it exists
     * @return the label opName
     */
    @Override
    public List<String> getLabelNames(INDArray idxs) {
        List<String> ret = new ArrayList<>();
        for (int i = 0; i < idxs.length(); i++) {
            ret.add(i, getLabelName(i));
        }
        return ret;

    }

    @Override
    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    /**
     * Get the feature matrix (inputs for the data)
     *
     * @return the feature matrix for the dataset
     */
    @Override
    public INDArray getFeatureMatrix() {
        return getFeatures();
    }


    /**
     * Organizes the dataset to minimize sampling error
     * while still allowing efficient batching.
     */
    @Override
    public void sortByLabel() {
        Map<Integer, Queue<DataSet>> map = new HashMap<>();
        List<DataSet> data = asList();
        int numLabels = numOutcomes();
        int examples = numExamples();
        for (DataSet d : data) {
            int label = d.outcome();
            Queue<DataSet> q = map.get(label);
            if (q == null) {
                q = new ArrayDeque<>();
                map.put(label, q);
            }
            q.add(d);
        }

        for (Map.Entry<Integer, Queue<DataSet>> label : map.entrySet()) {
            log.info("Label " + label + " has " + label.getValue().size() + " elements");
        }

        //ideal input splits: 1 of each label in each batch
        //after we run out of ideal batches: fall back to a new strategy
        boolean optimal = true;
        for (int i = 0; i < examples; i++) {
            if (optimal) {
                for (int j = 0; j < numLabels; j++) {
                    Queue<DataSet> q = map.get(j);
                    if (q == null) {
                        optimal = false;
                        break;
                    }
                    DataSet next = q.poll();
                    //add a row; go to next
                    if (next != null) {
                        addRow(next, i);
                        i++;
                    } else {
                        optimal = false;
                        break;
                    }
                }
            } else {
                DataSet add = null;
                for (Queue<DataSet> q : map.values()) {
                    if (!q.isEmpty()) {
                        add = q.poll();
                        break;
                    }
                }

                addRow(add, i);

            }


        }


    }


    @Override
    public void addRow(DataSet d, int i) {
        if (i > numExamples() || d == null)
            throw new IllegalArgumentException("Invalid index for adding a row");
        getFeatures().putRow(i, d.getFeatures());
        getLabels().putRow(i, d.getLabels());
    }


    private int getLabel(DataSet data) {
        Float f = data.getLabels().maxNumber().floatValue();
        return f.intValue();
    }


    @Override
    public INDArray exampleSums() {
        return getFeatures().sum(1);
    }

    @Override
    public INDArray exampleMaxs() {
        return getFeatures().max(1);
    }

    @Override
    public INDArray exampleMeans() {
        return getFeatures().mean(1);
    }


    /**
     * Sample without replacement and a random rng
     *
     * @param numSamples the number of samples to getFromOrigin
     * @return a sample data transform without replacement
     */
    @Override
    public DataSet sample(int numSamples) {
        return sample(numSamples, Nd4j.getRandom());
    }

    /**
     * Sample without replacement
     *
     * @param numSamples the number of samples to getFromOrigin
     * @param rng        the rng to use
     * @return the sampled dataset without replacement
     */
    @Override
    public DataSet sample(int numSamples, org.nd4j.linalg.api.rng.Random rng) {
        return sample(numSamples, rng, false);
    }

    /**
     * Sample a dataset numSamples times
     *
     * @param numSamples      the number of samples to getFromOrigin
     * @param withReplacement the rng to use
     * @return the sampled dataset without replacement
     */
    @Override
    public DataSet sample(int numSamples, boolean withReplacement) {
        return sample(numSamples, Nd4j.getRandom(), withReplacement);
    }

    /**
     * Sample a dataset
     *
     * @param numSamples      the number of samples to getFromOrigin
     * @param rng             the rng to use
     * @param withReplacement whether to allow duplicates (only tracked by example row number)
     * @return the sample dataset
     */
    @Override
    public DataSet sample(int numSamples, org.nd4j.linalg.api.rng.Random rng, boolean withReplacement) {
        INDArray examples = Nd4j.create(numSamples, getFeatures().columns());
        INDArray outcomes = Nd4j.create(numSamples, numOutcomes());
        Set<Integer> added = new HashSet<>();
        for (int i = 0; i < numSamples; i++) {
            int picked = rng.nextInt(numExamples());
            if (!withReplacement)
                while (added.contains(picked))
                    picked = rng.nextInt(numExamples());


            examples.putRow(i, get(picked).getFeatures());
            outcomes.putRow(i, get(picked).getLabels());

        }
        return new DataSet(examples, outcomes);

    }

    @Override
    public void roundToTheNearest(int roundTo) {
        for (int i = 0; i < getFeatures().length(); i++) {
            double curr = (double) getFeatures().getScalar(i).element();
            getFeatures().put(i, Nd4j.scalar(MathUtils.roundDouble(curr, roundTo)));
        }
    }

    @Override
    public int numOutcomes() {
        // FIXME: int cast
        return (int) getLabels().size(1);
    }

    @Override
    public int numExamples() {
        // FIXME: int cast
        if (getFeatureMatrix() != null)
            return (int) getFeatureMatrix().size(0);
        else if (getLabels() != null)
            return (int) getLabels().size(0);
        return 0;
    }


    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        if (features != null && labels != null) {
            builder.append("===========INPUT===================\n")
                            .append(getFeatures().toString().replaceAll(";", "\n"))
                            .append("\n=================OUTPUT==================\n")
                            .append(getLabels().toString().replaceAll(";", "\n"));
            if (featuresMask != null) {
                builder.append("\n===========INPUT MASK===================\n")
                                .append(getFeaturesMaskArray().toString().replaceAll(";", "\n"));
            }
            if (labelsMask != null) {
                builder.append("\n===========OUTPUT MASK===================\n")
                                .append(getLabelsMaskArray().toString().replaceAll(";", "\n"));
            }
            return builder.toString();
        } else {
            log.info("Features or labels are null values");
            return "";
        }
    }



    /**
     * Gets the optional label names
     *
     * @return
     */
    @Deprecated
    @Override
    public List<String> getLabelNames() {
        return labelNames;
    }


    /**
     * Gets the optional label names
     *
     * @return
     */
    @Override
    public List<String> getLabelNamesList() {
        return labelNames;
    }

    /**
     * Sets the label names, will throw an exception if the passed
     * in label names doesn't equal the number of outcomes
     *
     * @param labelNames the label names to use
     */
    @Override
    public void setLabelNames(List<String> labelNames) {
        this.labelNames = labelNames;
    }

    /**
     * Optional column names of the data transform, this is mainly used
     * for interpeting what columns are in the dataset
     *
     * @return
     */
    @Deprecated
    @Override
    public List<String> getColumnNames() {
        return columnNames;
    }

    /**
     * Sets the column names, will throw an exception if the column names
     * don't match the number of columns
     *
     * @param columnNames
     */
    @Deprecated
    @Override
    public void setColumnNames(List<String> columnNames) {
        this.columnNames = columnNames;
    }

    @Override
    public SplitTestAndTrain splitTestAndTrain(double fractionTrain) {
        Preconditions.checkArgument(fractionTrain > 0.0 && fractionTrain < 1.0,
                "Train fraction must be > 0.0 and < 1.0 - got %s", fractionTrain);
        int numTrain = (int) (fractionTrain * numExamples());
        if (numTrain <= 0)
            numTrain = 1;
        return splitTestAndTrain(numTrain);
    }


    @Override
    public Iterator<DataSet> iterator() {
        return asList().iterator();
    }

    @Override
    public INDArray getFeaturesMaskArray() {
        return featuresMask;
    }

    @Override
    public void setFeaturesMaskArray(INDArray featuresMask) {
        this.featuresMask = featuresMask;
    }

    @Override
    public INDArray getLabelsMaskArray() {
        return labelsMask;
    }

    @Override
    public void setLabelsMaskArray(INDArray labelsMask) {
        this.labelsMask = labelsMask;
    }

    @Override
    public boolean hasMaskArrays() {
        return labelsMask != null || featuresMask != null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof DataSet))
            return false;

        DataSet d = (DataSet) o;

        if (!equalOrBothNull(features, d.features))
            return false;
        if (!equalOrBothNull(labels, d.labels))
            return false;
        if (!equalOrBothNull(featuresMask, d.featuresMask))
            return false;
        return equalOrBothNull(labelsMask, d.labelsMask);
    }

    private static boolean equalOrBothNull(INDArray first, INDArray second) {
        if (first == null && second == null)
            return true; //Both are null: ok
        if (first == null || second == null)
            return false; //Only one is null, not both
        return first.equals(second);
    }

    @Override
    public int hashCode() {
        int result = getFeatures() != null ? getFeatures().hashCode() : 0;
        result = 31 * result + (getLabels() != null ? getLabels().hashCode() : 0);
        result = 31 * result + (getFeaturesMaskArray() != null ? getFeaturesMaskArray().hashCode() : 0);
        result = 31 * result + (getLabelsMaskArray() != null ? getLabelsMaskArray().hashCode() : 0);
        return result;
    }

    /**
     * This method returns memory used by this DataSet
     *
     * @return
     */
    @Override
    public long getMemoryFootprint() {
        long reqMem = features.lengthLong() * Nd4j.sizeOfDataType();
        reqMem += labels == null ? 0 : labels.lengthLong() * Nd4j.sizeOfDataType();
        reqMem += featuresMask == null ? 0 : featuresMask.lengthLong() * Nd4j.sizeOfDataType();
        reqMem += labelsMask == null ? 0 : labelsMask.lengthLong() * Nd4j.sizeOfDataType();

        return reqMem;
    }

    /**
     * This method migrates this DataSet into current Workspace (if any)
     */
    @Override
    public void migrate() {
        if (Nd4j.getMemoryManager().getCurrentWorkspace() != null) {
            if (features != null)
                features = features.migrate();

            if (labels != null)
                labels = labels.migrate();

            if (featuresMask != null)
                featuresMask = featuresMask.migrate();

            if (labelsMask != null)
                labelsMask = labelsMask.migrate();
        }
    }

    /**
     * This method migrates this DataSet into current Workspace (if any)
     */
    @Override
    public void detach() {
        if (features != null)
            features = features.detach();

        if (labels != null)
            labels = labels.detach();

        if (featuresMask != null)
            featuresMask = featuresMask.detach();

        if (labelsMask != null)
            labelsMask = labelsMask.detach();
    }

    @Override
    public boolean isEmpty() {
        return features == null && labels == null && featuresMask == null && labelsMask == null;
    }


}
