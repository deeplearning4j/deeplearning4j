/*-
 *  * Copyright 2016 Skymind,Inc.
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
 */

package org.deeplearning4j.spark.datavec;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.io.WritableConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.StringSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Turn a string in to a dataset based on
 * a record reader
 *
 * @author Adam Gibson
 */
public class RecordReaderFunction implements Function<String, DataSet> {
    private RecordReader recordReader;
    private int labelIndex = -1;
    private int numPossibleLabels = -1;
    private WritableConverter converter;

    public RecordReaderFunction(RecordReader recordReader, int labelIndex, int numPossibleLabels,
                    WritableConverter converter) {
        this.recordReader = recordReader;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.converter = converter;

    }

    public RecordReaderFunction(RecordReader recordReader, int labelIndex, int numPossibleLabels) {
        this(recordReader, labelIndex, numPossibleLabels, null);
    }

    @Override
    public DataSet call(String v1) throws Exception {
        recordReader.initialize(new StringSplit(v1));
        List<DataSet> dataSets = new ArrayList<>();
        List<Writable> record = recordReader.next();
        List<Writable> currList;
        if (record instanceof List)
            currList = (List<Writable>) record;
        else
            currList = new ArrayList<>(record);

        INDArray label = null;
        INDArray featureVector = Nd4j.create(labelIndex >= 0 ? currList.size() - 1 : currList.size());
        int count = 0;
        for (int j = 0; j < currList.size(); j++) {
            if (labelIndex >= 0 && j == labelIndex) {
                if (numPossibleLabels < 1)
                    throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
                Writable current = currList.get(j);
                if (converter != null)
                    current = converter.convert(current);
                label = FeatureUtil.toOutcomeVector(Double.valueOf(current.toString()).intValue(), numPossibleLabels);
            } else {
                Writable current = currList.get(j);
                featureVector.putScalar(count++, Double.valueOf(current.toString()));
            }
        }

        dataSets.add(new DataSet(featureVector, labelIndex >= 0 ? label : featureVector));



        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();
        for (DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }


        DataSet ret = new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])),
                        Nd4j.vstack(labels.toArray(new INDArray[0])));
        return ret;
    }
}
