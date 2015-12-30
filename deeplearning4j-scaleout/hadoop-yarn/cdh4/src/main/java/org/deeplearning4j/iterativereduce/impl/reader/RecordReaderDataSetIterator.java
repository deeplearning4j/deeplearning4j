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

package org.deeplearning4j.iterativereduce.impl.reader;


import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.RecordReader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetPreProcessor;
import org.deeplearning4j.iterativereduce.runtime.io.WritableConverter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A DataSetIterator that uses a record reader for
 * creating datasets from records
 *
 * @author Adam Gibson
 */
public class RecordReaderDataSetIterator implements DataSetIterator {
    private RecordReader recordReader;
    private WritableConverter converter;
    private int batchSize = 10;
    private int labelIndex = -1;
    private int numPossibleLabels = -1;

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,int labelIndex,int numPossibleLabels) {
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
    }

    public RecordReaderDataSetIterator(RecordReader recordReader,WritableConverter converter) {
        this(recordReader, converter, 10,-1,-1);
    }


    public RecordReaderDataSetIterator(RecordReader recordReader,WritableConverter converter,int labelIndex,int numPossibleLabels) {
        this(recordReader, converter, 10,labelIndex,numPossibleLabels);
    }



    @Override
    public DataSet next(int num) {
        List<DataSet> dataSets = new ArrayList<>();
        for(int i = 0; i < num; i++) {
            Collection<Writable> record = null;
            try {
                record = (Collection<Writable>) recordReader.getCurrentValue();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if(record == null)
                throw new IllegalStateException("No data found");

            List<Writable> currList;
            if(record instanceof List)
                currList = (List<Writable>) record;
            else
                currList = new ArrayList<>(record);

            INDArray label = null;
            INDArray featureVector = Nd4j.create(labelIndex >= 0 ? currList.size() - 1 : currList.size());
            int count = 0;
            for(int j = 0; j < currList.size(); j++) {
                if(labelIndex >= 0 && j == labelIndex) {
                    if(numPossibleLabels < 1)
                        throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
                    Writable current = currList.get(j);
                    if(converter != null)
                        current = converter.convert(current);
                    label = FeatureUtil.toOutcomeVector(Double.valueOf(current.toString()).intValue(),numPossibleLabels);
                }
                else {
                    Writable current = currList.get(j);

                    featureVector.putScalar(count++,Double.valueOf(current.toString()));
                }
            }

            dataSets.add(new DataSet(featureVector,labelIndex >= 0 ? label : featureVector));


        }

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();
        for(DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }

        return new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])),Nd4j.vstack(labels.toArray(new INDArray[0])));
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();

    }

    @Override
    public void reset() {

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

    }

    @Override
    public List<String> getLabels() {
        return null;
    }


    @Override
    public boolean hasNext() {
        try {
            return recordReader.getProgress() < 1;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();

    }
}
