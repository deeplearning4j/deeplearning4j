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
 */

package org.deeplearning4j.datasets.datavec;


import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.WritableConverter;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataComposableMap;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.ConcatenatingRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


/**
 * Record reader dataset iterator. Takes a DataVec {@link RecordReader} as input, and handles the conversion to ND4J
 * DataSet objects as well as producing minibatches from individual records.<br>
 * <br>
 * Multiple constructors are available, though a {@link Builder} class is also available.<br>
 * <br>
 * Example 1: Image classification, batch size 32, 10 classes<br>
 * <pre>
 * {@code RecordReader rr = new ImageRecordReader(28,28,3); //28x28 RGB images
 *  rr.initialize(new FileSplit(new File("/path/to/directory")));
 *
 *  DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, 32)
 *       //Label index (first arg): Always value 1 when using ImageRecordReader. For CSV etc: use index of the column
 *       //  that contains the label (should contain an integer value, 0 to nClasses-1 inclusive). Column indexes start
 *       // at 0. Number of classes (second arg): number of label classes (i.e., 10 for MNIST - 10 digits)
 *       .classification(1, nClasses)
 *       .preProcessor(new ImagePreProcessingScaler())      //For normalization of image values 0-255 to 0-1
 *       .build()
 * }
 * </pre>
 * <br>
 * <br>
 * Example 2: Multi-output regression from CSV, batch size 128<br>
 * <pre>
 * {@code RecordReader rr = new CsvRecordReader(0, ','); //Skip 0 header lines, comma separated
 *  rr.initialize(new FileSplit(new File("/path/to/myCsv.txt")));
 *
 *  DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, 128)
 *       //Specify the columns that the regression labels/targets appear in. Note that all other columns will be
 *       // treated as features. Columns indexes start at 0
 *       .regression(labelColFrom, labelColTo)
 *       .build()
 * }
 * </pre>
 * @author Adam Gibson
 */
@Slf4j
public class RecordReaderDataSetIterator implements DataSetIterator {
    private static final String READER_KEY = "reader";

    protected RecordReader recordReader;
    protected WritableConverter converter;
    protected int batchSize = 10;
    protected int maxNumBatches = -1;
    protected int batchNum = 0;
    protected int labelIndex = -1;
    protected int labelIndexTo = -1;
    protected int numPossibleLabels = -1;
    protected Iterator<List<Writable>> sequenceIter;
    protected DataSet last;
    protected boolean useCurrent = false;
    protected boolean regression = false;
    @Getter
    protected DataSetPreProcessor preProcessor;

    @Getter
    private boolean collectMetaData = false;

    private RecordReaderMultiDataSetIterator underlying;
    private boolean underlyingIsDisjoint;

    /**
     * Constructor for classification, where:<br>
     * (a) the label index is assumed to be the very last Writable/column, and<br>
     * (b) the number of classes is inferred from RecordReader.getLabels()<br>
     * Note that if RecordReader.getLabels() returns null, no output labels will be produced
     *
     * @param recordReader Record reader to use as the source of data
     * @param batchSize    Minibatch size, for each call of .next()
     */
    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize) {
        this(recordReader, new SelfWritableConverter(), batchSize, -1, -1,
                        recordReader.getLabels() == null ? -1 : recordReader.getLabels().size(), -1, false);
    }

    /**
     * Main constructor for classification. This will convert the input class index (at position labelIndex, with integer
     * values 0 to numPossibleLabels-1 inclusive) to the appropriate one-hot output/labels representation.
     *
     * @param recordReader         RecordReader: provides the source of the data
     * @param batchSize            Batch size (number of examples) for the output DataSet objects
     * @param labelIndex           Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
     * @param numPossibleLabels    Number of classes (possible labels) for classification
     */
    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex,
                    int numPossibleLabels) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, labelIndex, numPossibleLabels, -1, false);
    }

    /**
     * Constructor for classification, where the maximum number of returned batches is limited to the specified value
     *
     * @param recordReader      the recordreader to use
     * @param labelIndex        the index/column of the label (for classification)
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param maxNumBatches     The maximum number of batches to return between resets. Set to -1 to return all available data
     */
    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels,
                    int maxNumBatches) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, labelIndex, numPossibleLabels, maxNumBatches, false);
    }

    /**
     * Main constructor for multi-label regression (i.e., regression with multiple outputs)
     *
     * @param recordReader      RecordReader to get data from
     * @param labelIndexFrom    Index of the first regression target
     * @param labelIndexTo      Index of the last regression target, inclusive
     * @param batchSize         Minibatch size
     * @param regression        Require regression = true. Mainly included to avoid clashing with other constructors previously defined :/
     */
    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
                    boolean regression) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndexFrom, labelIndexTo, -1, -1, regression);
    }


    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param converter         Converter. May be null.
     * @param batchSize         Minibatch size - number of examples returned for each call of .next()
     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
     * @param labelIndexTo      only used if regression == true. The last index <i>inclusive</i> of the multi-output regression
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param maxNumBatches     Maximum number of batches to return
     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is the class it belongs to)
     */
    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                    int labelIndexFrom, int labelIndexTo, int numPossibleLabels, int maxNumBatches,
                    boolean regression) {
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.maxNumBatches = maxNumBatches;
        this.labelIndex = labelIndexFrom;
        this.labelIndexTo = labelIndexTo;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
    }


    protected RecordReaderDataSetIterator(Builder b){
        this.recordReader = b.recordReader;
        this.converter = b.converter;
        this.batchSize = b.batchSize;
        this.maxNumBatches = b.maxNumBatches;
        this.labelIndex = b.labelIndex;
        this.labelIndexTo = b.labelIndexTo;
        this.numPossibleLabels = b.numPossibleLabels;
        this.regression = b.regression;
        this.preProcessor = b.preProcessor;
    }

    /**
     * When set to true: metadata for  the current examples will be present in the returned DataSet.
     * Disabled by default.
     *
     * @param collectMetaData Whether to collect metadata or  not
     */
    public void setCollectMetaData(boolean collectMetaData) {
        if (underlying != null) {
            underlying.setCollectMetaData(collectMetaData);
        }
        this.collectMetaData = collectMetaData;
    }

    private void initializeUnderlying(){
        if (underlying == null) {
            Record next = recordReader.nextRecord();
            initializeUnderlying(next);
        }
    }

    private void initializeUnderlying(Record next) {
        int totalSize = next.getRecord().size();

        //allow people to specify label index as -1 and infer the last possible label
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = totalSize - 1;
            labelIndexTo = labelIndex;
        }

        if(recordReader.resetSupported()) {
            recordReader.reset();
        } else {
            //Hack around the fact that we need the first record to initialize the underlying RRMDSI, but can't reset
            // the original reader
            recordReader = new ConcatenatingRecordReader(
                    new CollectionRecordReader(Collections.singletonList(next.getRecord())),
                    recordReader);
        }

        RecordReaderMultiDataSetIterator.Builder builder = new RecordReaderMultiDataSetIterator.Builder(batchSize);
        if (recordReader instanceof SequenceRecordReader) {
            builder.addSequenceReader(READER_KEY, (SequenceRecordReader) recordReader);
        } else {
            builder.addReader(READER_KEY, recordReader);
        }

        if (regression) {
            builder.addOutput(READER_KEY, labelIndex, labelIndexTo);
        } else if (numPossibleLabels >= 1) {
            builder.addOutputOneHot(READER_KEY, labelIndex, numPossibleLabels);
        }

        //Inputs: assume to be all of the other writables
        //In general: can't assume label indices are all at the start or end (event though 99% of the time they are)
        //If they are: easy. If not: use 2 inputs in the underlying as a workaround, and concat them

        if (labelIndex >= 0 && (labelIndex == 0 || labelIndexTo == totalSize - 1)) {
            //Labels are first or last -> one input in underlying
            int inputFrom;
            int inputTo;
            if (labelIndex < 0) {
                //No label
                inputFrom = 0;
                inputTo = totalSize - 1;
            } else if (labelIndex == 0) {
                inputFrom = labelIndexTo + 1;
                inputTo = totalSize - 1;
            } else {
                inputFrom = 0;
                inputTo = labelIndex - 1;
            }

            builder.addInput(READER_KEY, inputFrom, inputTo);

            underlyingIsDisjoint = false;
        } else if (labelIndex >= 0) {
            //Multiple inputs
            int firstFrom = 0;
            int firstTo = labelIndex - 1;
            int secondFrom = labelIndexTo + 1;
            int secondTo = totalSize - 1;

            builder.addInput(READER_KEY, firstFrom, firstTo);
            builder.addInput(READER_KEY, secondFrom, secondTo);

            underlyingIsDisjoint = true;
        } else {
            //No labels - only features
            builder.addInput(READER_KEY);
            underlyingIsDisjoint = false;
        }


        underlying = builder.build();

        if (collectMetaData) {
            underlying.setCollectMetaData(true);
        }
    }

    private DataSet mdsToDataSet(MultiDataSet mds) {
        INDArray f;
        INDArray fm;
        if (underlyingIsDisjoint) {
            //Rare case: 2 input arrays -> concat
            INDArray f1 = getOrNull(mds.getFeatures(), 0);
            INDArray f2 = getOrNull(mds.getFeatures(), 1);
            fm = getOrNull(mds.getFeaturesMaskArrays(), 0); //Per-example masking only on the input -> same for both

            //Can assume 2d features here
            f = Nd4j.hstack(f1, f2);
        } else {
            //Standard case
            f = getOrNull(mds.getFeatures(), 0);
            fm = getOrNull(mds.getFeaturesMaskArrays(), 0);
        }

        INDArray l = getOrNull(mds.getLabels(), 0);
        INDArray lm = getOrNull(mds.getLabelsMaskArrays(), 0);

        DataSet ds = new DataSet(f, l, fm, lm);

        if (collectMetaData) {
            List<Serializable> temp = mds.getExampleMetaData();
            List<Serializable> temp2 = new ArrayList<>(temp.size());
            for (Serializable s : temp) {
                RecordMetaDataComposableMap m = (RecordMetaDataComposableMap) s;
                temp2.add(m.getMeta().get(READER_KEY));
            }
            ds.setExampleMetaData(temp2);
        }

        //Edge case, for backward compatibility:
        //If labelIdx == -1 && numPossibleLabels == -1 -> no labels -> set labels array to features array
        if (labelIndex == -1 && numPossibleLabels == -1 && ds.getLabels() == null) {
            ds.setLabels(ds.getFeatures());
        }

        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }

        return ds;
    }


    @Override
    public DataSet next(int num) {
        if (useCurrent) {
            useCurrent = false;
            if (preProcessor != null)
                preProcessor.preProcess(last);
            return last;
        }

        if (underlying == null) {
            initializeUnderlying();
        }


        batchNum++;
        return mdsToDataSet(underlying.next(num));
    }

    //Package private
    static INDArray getOrNull(INDArray[] arr, int idx) {
        if (arr == null || arr.length == 0) {
            return null;
        }
        return arr[idx];
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        } else
            return last.numInputs();
    }

    @Override
    public int totalOutcomes() {
        if (last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        } else
            return last.numOutcomes();
    }

    @Override
    public boolean resetSupported() {
        if(underlying == null){
            initializeUnderlying();
        }
        return underlying.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        batchNum = 0;
        if (underlying != null) {
            underlying.reset();
        }

        last = null;
        useCurrent = false;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(org.nd4j.linalg.dataset.api.DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public boolean hasNext() {
        return (((sequenceIter != null && sequenceIter.hasNext()) || recordReader.hasNext())
                        && (maxNumBatches < 0 || batchNum < maxNumBatches));
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return recordReader.getLabels();
    }

    /**
     * Load a single example to a DataSet, using the provided RecordMetaData.
     * Note that it is more efficient to load multiple instances at once, using {@link #loadFromMetaData(List)}
     *
     * @param recordMetaData RecordMetaData to load from. Should have been produced by the given record reader
     * @return DataSet with the specified example
     * @throws IOException If an error occurs during loading of the data
     */
    public DataSet loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData));
    }

    /**
     * Load a multiple examples to a DataSet, using the provided RecordMetaData instances.
     *
     * @param list List of RecordMetaData instances to load from. Should have been produced by the record reader provided
     *             to the RecordReaderDataSetIterator constructor
     * @return DataSet with the specified examples
     * @throws IOException If an error occurs during loading of the data
     */
    public DataSet loadFromMetaData(List<RecordMetaData> list) throws IOException {
        if (underlying == null) {
            Record r = recordReader.loadFromMetaData(list.get(0));
            initializeUnderlying(r);
        }

        //Convert back to composable:
        List<RecordMetaData> l = new ArrayList<>(list.size());
        for (RecordMetaData m : list) {
            l.add(new RecordMetaDataComposableMap(Collections.singletonMap(READER_KEY, m)));
        }
        MultiDataSet m = underlying.loadFromMetaData(l);

        return mdsToDataSet(m);
    }

    /**
     * Builder class for RecordReaderDataSetIterator
     */
    public static class Builder {

        protected RecordReader recordReader;
        protected WritableConverter converter;
        protected int batchSize;
        protected int maxNumBatches = -1;
        protected int labelIndex = -1;
        protected int labelIndexTo = -1;
        protected int numPossibleLabels = -1;
        protected boolean regression = false;
        protected DataSetPreProcessor preProcessor;
        private boolean collectMetaData = false;

        private boolean clOrRegCalled = false;

        /**
         *
         * @param rr        Underlying record reader to source data from
         * @param batchSize Batch size to use
         */
        public Builder(@NonNull RecordReader rr, int batchSize){
            this.recordReader = rr;
            this.batchSize = batchSize;
        }

        public Builder writableConverter(WritableConverter converter){
            this.converter = converter;
            return this;
        }

        /**
         * Optional argument, usually not used. If set, can be used to limit the maximum number of minibatches that
         * will be returned (between resets). If not set, will always return as many minibatches as there is data
         * available.
         *
         * @param maxNumBatches Maximum number of minibatches per epoch / reset
         */
        public Builder maxNumBatches(int maxNumBatches){
            this.maxNumBatches = maxNumBatches;
            return this;
        }

        /**
         * Use this for single output regression (i.e., 1 output/regression target)
         *
         * @param labelIndex Column index that contains the regression target (indexes start at 0)
         */
        public Builder regression(int labelIndex){
            return regression(labelIndex, labelIndex);
        }

        /**
         * Use this for multiple output regression (1 or more output/regression targets). Note that all regression
         * targets must be contiguous (i.e., positions x to y, without gaps)
         *
         * @param labelIndexFrom Column index of the first regression target (indexes start at 0)
         * @param labelIndexTo   Column index of the last regression target (inclusive)
         */
        public Builder regression(int labelIndexFrom, int labelIndexTo){
            this.labelIndex = labelIndexFrom;
            this.labelIndexTo = labelIndexTo;
            this.regression = true;
            clOrRegCalled = true;
            return this;
        }

        /**
         * Use this for classification
         *
         * @param labelIndex Index that contains the label index. Column (indexes start from 0) be an integer value,
         *                   and contain values 0 to numClasses-1
         * @param numClasses Number of label classes (i.e., number of categories/classes in the dataset)
         */
        public Builder classification(int labelIndex, int numClasses){
            this.labelIndex = labelIndex;
            this.labelIndexTo = labelIndex;
            this.numPossibleLabels = numClasses;
            this.regression = false;
            clOrRegCalled = true;
            return this;
        }

        /**
         * Optional arg. Allows the preprocessor to be set
         * @param preProcessor Preprocessor to use
         */
        public Builder preProcessor(DataSetPreProcessor preProcessor){
            this.preProcessor = preProcessor;
            return this;
        }

        /**
         * When set to true: metadata for  the current examples will be present in the returned DataSet.
         * Disabled by default.
         *
         * @param collectMetaData Whether metadata should be collected or not
         */
        public Builder collectMetaData(boolean collectMetaData){
            this.collectMetaData = collectMetaData;
            return this;
        }

        public RecordReaderDataSetIterator build(){
            return new RecordReaderDataSetIterator(this);
        }

    }
}
