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
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.WritableConverter;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataComposableMap;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
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
 * Record reader dataset iterator
 *
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

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize) {
        this(recordReader, converter, batchSize, -1,
                        recordReader.getLabels() == null ? -1 : recordReader.getLabels().size());
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize) {
        this(recordReader, new SelfWritableConverter(), batchSize, -1,
                        recordReader.getLabels() == null ? -1 : recordReader.getLabels().size());
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
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                    int labelIndex, int numPossibleLabels, boolean regression) {
        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, regression);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                    int labelIndex, int numPossibleLabels) {
        this(recordReader, converter, batchSize, labelIndex, numPossibleLabels, -1, false);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels,
                    int maxNumBatches) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, maxNumBatches, false);
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


    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,
                    int labelIndex, int numPossibleLabels, int maxNumBatches, boolean regression) {
        this(recordReader, converter, batchSize, labelIndex, labelIndex, numPossibleLabels, maxNumBatches, regression);
    }


    /**
     * Main constructor
     *
     * @param recordReader      the recordreader to use
     * @param converter         the batch size
     * @param maxNumBatches     Maximum number of batches to return
     * @param labelIndexFrom    the index of the label (for classification), or the first index of the labels for multi-output regression
     * @param labelIndexTo      only used if regression == true. The last index _inclusive_ of the multi-output regression
     * @param numPossibleLabels the number of possible labels for classification. Not used if regression == true
     * @param regression        if true: regression. If false: classification (assume labelIndexFrom is a
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

    private void initializeUnderlying(Record next) {
        int totalSize = next.getRecord().size();

        //allow people to specify label index as -1 and infer the last possible label
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = totalSize - 1;
        }

        recordReader.reset();

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
            Record next = recordReader.nextRecord();
            initializeUnderlying(next);
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
        return true;
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
}
