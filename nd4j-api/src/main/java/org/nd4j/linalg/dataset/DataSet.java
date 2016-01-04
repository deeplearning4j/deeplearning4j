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
 *
 */

package org.nd4j.linalg.dataset;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.io.*;
import java.util.*;


/**
 * A data transform (example/outcome pairs)
 * The outcomes are specifically for neural network encoding such that
 * any labels that are considered true are 1s. The rest are zeros.
 *
 * @author Adam Gibson
 */
public class DataSet implements org.nd4j.linalg.dataset.api.DataSet {

    private static final long serialVersionUID = 1935520764586513365L;
    private static Logger log = LoggerFactory.getLogger(DataSet.class);
    private List<String> columnNames = new ArrayList<>();
    private List<String> labelNames = new ArrayList<>();
    private INDArray features, labels;
    private String id = UUID.randomUUID().toString();
    private INDArray featuresMask;
    private INDArray labelsMask;

    public DataSet() {
        this(Nd4j.zeros(new int[]{1,1}), Nd4j.zeros(new int[]{1,1}));
    }




    /**
     * Creates a dataset with the specified input matrix and labels
     *
     * @param first  the feature matrix
     * @param second the labels (these should be binarized label matrices such that the specified label
     *               has a value of 1 in the desired column with the label)
     */
    public DataSet(INDArray first, INDArray second) {
        this(first,second,null,null);
    }

    /**Create a dataset with the specified input INDArray and labels (output) INDArray, plus (optionally) mask arrays
     * for the features and labels
     * @param features Features (input)
     * @param labels Labels (output)
     * @param featuresMask Mask array for features, may be null
     * @param labelsMask Mask array for labels, may be null
     */
    public DataSet(INDArray features, INDArray labels, INDArray featuresMask, INDArray labelsMask) {
        if (features.size(0) != labels.size(0))
            throw new IllegalStateException("Invalid data transform; features and labels do not have equal rows. First was " + features.size(0) + " labels was " + labels.size(0));
        this.features = features;
        this.labels = labels;
        this.featuresMask = featuresMask;
        this.labelsMask = labelsMask;
    }

    /**
     * Returns a single dataset
     *
     * @return an empty dataset with 2 1x1 zero matrices
     */
    public static DataSet empty() {
        return new DataSet(Nd4j.zeros(new int[]{1,1}), Nd4j.zeros(new int[]{1,1}));
    }
    /**
     * Merge the list of datasets in to one list.
     * All the rows are merged in to one dataset
     *
     * @param data the data to merge
     * @param clone whether to clone the data
     *              or use a reference
     * @return a single dataset
     */
    public static DataSet merge(List<DataSet> data,boolean clone) {
        if (data.isEmpty())
            throw new IllegalArgumentException("Unable to merge empty dataset");
        DataSet first = data.get(0);
        if(first.getFeatures().rank() == 3 && first.getLabels().rank() == 3 ){
            return mergeTimeSeries(data);
        }

        int numExamples = totalExamples(data);
        INDArray in = Nd4j.create(numExamples, first.getFeatures().columns());
        INDArray out = Nd4j.create(numExamples, first.getLabels().columns());
        int count = 0;

        for (int i = 0; i < data.size(); i++) {
            DataSet d1 = data.get(i);
            for (int j = 0; j < d1.numExamples(); j++) {
                DataSet example = d1.get(j);
                in.putRow(count, clone ? example.getFeatures().dup() : example.getFeatures());
                out.putRow(count, clone ? example.getLabels().dup() : example.getLabels());
                count++;
            }


        }
        return new DataSet(in, out);
    }

    private static DataSet mergeTimeSeries(List<DataSet> data){
        if(data.size() == 1) return data.get(0);

        //Complications with time series:
        //(a) They may have different lengths (if so: need input + output masking arrays)
        //(b) Even if they are all the same length, they may have masking arrays (if so: merge the masking arrays too)

        DataSet first = data.get(0);
        int firstLength = first.getFeatureMatrix().size(2);
        int maxLength = firstLength;

        boolean hasInputMask = false;
        boolean hasOutputMask = false;
        boolean lengthsDiffer = false;
        for(DataSet ds : data){
            int thisLength = ds.getFeatureMatrix().size(2);
            maxLength = Math.max(maxLength,thisLength);
            if( thisLength != firstLength ) lengthsDiffer = true;
            if( ds.getFeaturesMaskArray() != null ) hasInputMask = true;
            if( ds.getLabelsMaskArray() != null ) hasOutputMask = true;
        }

        boolean needInputMask = hasInputMask || lengthsDiffer;
        boolean needOutputMask = hasOutputMask || lengthsDiffer;

        int numExamples = totalExamples(data);
        int nIn = first.getFeatureMatrix().size(1);
        int nOut = first.getLabels().size(1);

        INDArray in = Nd4j.create(numExamples, nIn, maxLength);
        INDArray out = Nd4j.create(numExamples, nOut, maxLength);

        INDArray featuresMask = (needInputMask ? Nd4j.create(numExamples,maxLength) : null);
        INDArray labelsMask = (needOutputMask ? Nd4j.create(numExamples,maxLength) : null);

        int rowCount = 0;

        if(!lengthsDiffer && !needInputMask && !needOutputMask) {
            //Simplest case: no masking arrays, all same length
            for (DataSet ds : data) {
                INDArray f = ds.getFeatures();
                INDArray l = ds.getLabels();
                int nEx = f.size(0);

                in.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.all(), NDArrayIndex.all()).assign(f);
                out.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.all(), NDArrayIndex.all()).assign(l);
                rowCount += nEx;
            }
        } else {
            for (DataSet ds : data) {
                INDArray f = ds.getFeatures();
                INDArray l = ds.getLabels();
                int nEx = f.size(0);
                int thisLength = f.size(2);

                in.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.all(), NDArrayIndex.interval(0,thisLength)).assign(f);
                out.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.all(), NDArrayIndex.interval(0, thisLength)).assign(l);

                if(needInputMask){
                    INDArray inputMask = ds.getFeaturesMaskArray();
                    if(inputMask != null){
                        //Combine the input mask for this dataset with the overall (merged) input mask
                        // Not necessary to pad to give same overall length, as features mask is initialized as 0
                        //inputMask shape: [nEx,thisLength]
                        featuresMask.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.interval(0,thisLength))
                                .assign(inputMask);
                    } else {
                        //No input mask -> all feature values are present for entire length of the time series
                        featuresMask.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.interval(0,thisLength))
                                .assign(1.0);
                    }
                }

                if(needOutputMask){
                    INDArray outputMask = ds.getLabelsMaskArray();
                    if(outputMask != null){
                        //Combine the output mask for this dataset with the overall (merged) output mask
                        // Not necessary to pad to give same overall length, as labels mask is initialized as 0
                        //outputMask shape: [nEx,thisLength]
                        labelsMask.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.interval(0,thisLength))
                                .assign(outputMask);
                    } else {
                        //No output mask -> all feature values are present for entire length of the time series
                        labelsMask.get(NDArrayIndex.interval(rowCount, rowCount + nEx), NDArrayIndex.interval(0,thisLength))
                                .assign(1.0);
                    }
                }

                rowCount += nEx;
            }
        }

        return new DataSet(in, out, featuresMask, labelsMask);
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
        return merge(data,false);
    }

    private static int totalExamples(Collection<DataSet> coll) {
        int count = 0;
        for (DataSet d : coll)
            count += d.numExamples();
        return count;
    }

    @Override
    public org.nd4j.linalg.dataset.api.DataSet getRange(int from, int to) {
        return new DataSet(features.get(NDArrayIndex.interval(from,to)),labels.get(NDArrayIndex.interval(from,to)));
    }

    @Override
    public void load(File from) {
        try {
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(from));
            DataInputStream dis = new DataInputStream(bis);
            features = Nd4j.read(dis);
            labels = Nd4j.read(dis);
            dis.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void save(File to) {
        try {
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to));
            DataOutputStream dis = new DataOutputStream(bos);
            Nd4j.write(getFeatureMatrix(),dis);
            Nd4j.write(getLabels(),dis);
            dis.flush();
            dis.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public DataSetIterator iterateWithMiniBatches() {
        return null;
    }

    @Override
    public String id() {
        return id;
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
        int nTensors = labels.tensorssAlongDimension(1);
        for( int i=0; i<nTensors; i++ ) {
            INDArray row = labels.tensorAlongDimension(i, 1);
            int maxIdx = Nd4j.getBlasWrapper().iamax(row);
            if (maxIdx < 0)
                throw new IllegalStateException("Please check the iamax implementation for " + Nd4j.getBlasWrapper().getClass().getName());
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
        DataSet ret = new DataSet(getFeatures().reshape(new int[]{rows, cols}), getLabels());
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
        //note here we use the same seed with different random objects guaranteeing same order
        long seed = System.currentTimeMillis();
        Nd4j.shuffle(getFeatureMatrix(),new Random(seed),1);
        Nd4j.shuffle(getLabels(),new Random(seed),1);
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
        FeatureUtil.normalizeMatrix(getFeatures());
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
     * Subtract by the column means and divide by the standard deviation
     */
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
        return getFeatures().columns();
    }

    @Override
    public void validate() {
        if (getFeatures().size(0) != getLabels().size(0))
            throw new IllegalStateException("Invalid dataset");
    }

    @Override
    public int outcome() {
        if (this.numExamples() > 1)
            throw new IllegalStateException("Unable to derive outcome for dataset greater than one row");
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
        if(i == 0 && numExamples() == 1)
            return this;
        return new DataSet(getFeatures().getRow(i), getLabels().getRow(i));
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
        for(List<DataSet> splitBatch : Lists.partition(asList(), num)) {
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
        for (int i = 0; i < numExamples(); i++) {
            list.add(new DataSet(getFeatures().getRow(i), getLabels().getRow(i)));
        }
        return list;
    }

    /**
     * Splits a dataset in to test and train
     *
     * @param numHoldout the number to hold out for training
     * @return the pair of datasets for the train test split
     */
    @Override
    public SplitTestAndTrain splitTestAndTrain(int numHoldout, Random rng) {
        if (numHoldout >= numExamples())
            throw new IllegalArgumentException("Unable to split on size larger than the number of rows");
        DataSet first = new DataSet(getFeatureMatrix().get(NDArrayIndex.interval(0,numHoldout)),getLabels().get(NDArrayIndex.interval(0,numHoldout)));
        DataSet second = new DataSet(getFeatureMatrix().get(NDArrayIndex.interval(numHoldout,numExamples())),getLabels().get(NDArrayIndex.interval(numHoldout,numExamples())));
        return new SplitTestAndTrain(first, second);
    }

    @Override
    public SplitTestAndTrain splitTestAndTrain(int numHoldout) {
        return splitTestAndTrain(numHoldout, new Random());
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
        return sample(numSamples,Nd4j.getRandom());
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
        return getLabels().columns();
    }

    @Override
    public int numExamples() {
        return getFeatures().size(0);
    }


    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("===========INPUT===================\n")
                .append(getFeatures().toString().replaceAll(";", "\n"))
                .append("\n=================OUTPUT==================\n")
                .append(getLabels().toString().replaceAll(";", "\n"));
        return builder.toString();
    }


    /**
     * Gets the optional label names
     *
     * @return
     */
    @Override
    public List<String> getLabelNames() {
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
    @Override
    public void setColumnNames(List<String> columnNames) {
        this.columnNames = columnNames;
    }

    @Override
    public SplitTestAndTrain splitTestAndTrain(double percentTrain) {
        int numPercent = (int) (percentTrain * numExamples());
        return splitTestAndTrain(numPercent);
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
        if (this == o) return true;
        if (!(o instanceof DataSet)) return false;

        DataSet dataSet = (DataSet) o;

        if (getFeatures() != null ? !getFeatures().equals(dataSet.getFeatures()) : dataSet.getFeatures() != null)
            return false;
        return !(getLabels() != null ? !getLabels().equals(dataSet.getLabels()) : dataSet.getLabels() != null);

    }

    @Override
    public int hashCode() {
        int result = getFeatures() != null ? getFeatures().hashCode() : 0;
        result = 31 * result + (getLabels() != null ? getLabels().hashCode() : 0);
        return result;
    }
}
