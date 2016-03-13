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
 */

package org.deeplearning4j.datasets.canova;

import com.google.common.annotations.VisibleForTesting;
import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.SelfWritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.*;


/**
 * Record reader dataset iterator
 *
 * @author Adam Gibson
 */
public class RecordReaderDataSetIterator implements DataSetIterator {
    protected RecordReader recordReader;
    protected WritableConverter converter;
    protected int batchSize = 10;
    protected int maxNumBatches = -1;
    protected int batchNum = 0;
    protected int labelIndex = -1;
    protected int numPossibleLabels = -1;
    protected boolean notOvershot = true;
    protected Iterator<Collection<Writable>> sequenceIter;
    protected DataSet last;
    protected boolean useCurrent = false;
    protected boolean regression = false;
    protected DataSetPreProcessor preProcessor;


    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize) {
        this(recordReader, new SelfWritableConverter(), batchSize, -1, -1);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels, int maxNumBatches) {
        this(recordReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, maxNumBatches, false);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader) {
        this(recordReader, new SelfWritableConverter());
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, int labelIndex, int numPossibleLabels) {
        this(recordReader, new SelfWritableConverter(), 10, labelIndex, numPossibleLabels);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels, boolean regression) {
        this(recordReader,converter,batchSize,labelIndex,numPossibleLabels, -1, regression);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels) {
        this(recordReader,converter,batchSize,labelIndex,numPossibleLabels, -1, false);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter) {
        this(recordReader, converter, 10, -1, -1);
    }

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int labelIndex, int numPossibleLabels) {
        this(recordReader, converter, 10, labelIndex, numPossibleLabels);
    }


    /**
     * Main constructor
     * @param recordReader the recordreader to use
     * @param converter the batch size
     * @param maxNumBatches
     * @param labelIndex the index of the label
     * @param numPossibleLabels the number of possible labels for classification
     * @param regression
     */
    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize, int labelIndex, int numPossibleLabels, int maxNumBatches, boolean regression) {
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.maxNumBatches = maxNumBatches;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
    }


    @Override
    public DataSet next(int num) {
        if(useCurrent) {
            useCurrent = false;
            if(preProcessor != null) preProcessor.preProcess(last);
            return last;
        }

        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            if (!hasNext())
                break;
            if (recordReader instanceof SequenceRecordReader) {
                if(sequenceIter == null || !sequenceIter.hasNext()) {
                    Collection<Collection<Writable>> sequenceRecord = ((SequenceRecordReader) recordReader).sequenceRecord();
                    sequenceIter = sequenceRecord.iterator();
                }
                Collection<Writable> record = sequenceIter.next();
                dataSets.add(getDataSet(record));
            }

            else {
                Collection<Writable> record = recordReader.next();
                dataSets.add(getDataSet(record));
            }
        }
        batchNum++;

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for (DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }


        if(inputs.isEmpty()) {
            notOvershot = false;
            return last;
        } else if (maxNumBatches > -1 && batchNum >= maxNumBatches){
            notOvershot = false;
        }

        DataSet ret =  new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])), Nd4j.vstack(labels.toArray(new INDArray[0])));
        last = ret;
        if(preProcessor != null) preProcessor.preProcess(ret);
        return ret;
    }


    private DataSet getDataSet(Collection<Writable> record) {
        List<Writable> currList;
        if (record instanceof List)
            currList = (List<Writable>) record;
        else
            currList = new ArrayList<>(record);

        //allow people to specify label index as -1 and infer the last possible label
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = record.size() - 1;
        }

        INDArray label = null;
        INDArray featureVector = Nd4j.create(labelIndex >= 0 ? currList.size()-1 : currList.size());
        int featureCount = 0;
        for (int j = 0; j < currList.size(); j++) {
            Writable current = currList.get(j);
            if (current.toString().isEmpty())
                continue;
            if (labelIndex >= 0 && j == labelIndex) {
                if (converter != null)
                    try {
                        current = converter.convert(current);
                    } catch (WritableConverterException e) {
                        e.printStackTrace();
                    }
                if (numPossibleLabels < 1)
                    throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
                if (regression) {
                    label = Nd4j.scalar(current.toDouble());
                } else {
                    int curr = current.toInt();
                    if (curr >= numPossibleLabels)
                        curr--;
                    label = FeatureUtil.toOutcomeVector(curr, numPossibleLabels);
                }
            } else {
                featureVector.putScalar(featureCount++, current.toDouble());
            }
        }

        return new DataSet(featureVector, labelIndex >= 0 ? label : featureVector);
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        if(last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numInputs();
        }
        else
            return last.numInputs();

    }

    @Override
    public int totalOutcomes() {
        if(last == null) {
            DataSet next = next();
            last = next;
            useCurrent = true;
            return next.numOutcomes();
        }
        else
            return last.numOutcomes();


    }

    @Override
    public void reset() {
        if (recordReader instanceof RecordReader){
            batchNum = 0;
            notOvershot = true;
            recordReader.reset();
        }
        else if (recordReader instanceof SequenceRecordReader)
            throw new UnsupportedOperationException("Reset not supported for SequenceRecordReader type.");
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
        return (recordReader.hasNext() && notOvershot);
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
    public List<String> getLabels(){
        return recordReader.getLabels();
    }

}
