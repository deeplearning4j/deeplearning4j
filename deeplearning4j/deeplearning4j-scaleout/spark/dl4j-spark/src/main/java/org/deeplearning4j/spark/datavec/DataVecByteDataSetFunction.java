/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.datavec;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import scala.Tuple2;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 */
public class DataVecByteDataSetFunction implements PairFunction<Tuple2<Text, BytesWritable>, Double, DataSet> {

    private int labelIndex = 0;
    private int numPossibleLabels;
    private int byteFileLen;
    private int batchSize;
    private int numExamples;
    private boolean regression = false;
    private DataSetPreProcessor preProcessor;

    public DataVecByteDataSetFunction(int labelIndex, int numPossibleLabels, int batchSize, int byteFileLen) {
        this(labelIndex, numPossibleLabels, batchSize, byteFileLen, false, null);
    }

    public DataVecByteDataSetFunction(int labelIndex, int numPossibleLabels, int batchSize, int byteFileLen,
                    boolean regression) {
        this(labelIndex, numPossibleLabels, batchSize, byteFileLen, regression, null);
    }

    /**
     * @param labelIndex Index of the label column
     * @param numPossibleLabels Number of classes for classification  (not used if regression = true)
     * @param batchSize size of examples in DataSet. Pass in total examples if including all
     * @param byteFileLen number of bytes per individual file
     * @param regression False for classification, true for regression
     * @param preProcessor DataSetPreprocessor (may be null)
     */
    public DataVecByteDataSetFunction(int labelIndex, int numPossibleLabels, int batchSize, int byteFileLen,
                    boolean regression, DataSetPreProcessor preProcessor) {
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.batchSize = batchSize;
        this.byteFileLen = byteFileLen;
        this.regression = regression;
        this.preProcessor = preProcessor;

    }

    @Override
    public Tuple2<Double, DataSet> call(Tuple2<Text, BytesWritable> inputTuple) throws Exception {
        int lenFeatureVector = 0;

        if (numPossibleLabels >= 1) {
            lenFeatureVector = byteFileLen - 1;
            if (labelIndex < 0)
                labelIndex = byteFileLen - 1;
        }

        InputStream inputStream = new DataInputStream(new ByteArrayInputStream(inputTuple._2().getBytes()));

        int batchNumCount = 0;
        byte[] byteFeature = new byte[byteFileLen];
        List<DataSet> dataSets = new ArrayList<>();
        INDArray label;
        int featureCount;

        try {
            INDArray featureVector = Nd4j.create(lenFeatureVector);
            while ((inputStream.read(byteFeature)) != -1 && batchNumCount != batchSize) {
                featureCount = 0;
                label = FeatureUtil.toOutcomeVector(byteFeature[labelIndex], numPossibleLabels);
                for (int j = 1; j <= featureVector.length(); j++)
                    featureVector.putScalar(featureCount++, byteFeature[j]);
                dataSets.add(new DataSet(featureVector, label));
                batchNumCount++;
                byteFeature = new byte[byteFileLen];
                featureVector = Nd4j.create(lenFeatureVector);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for (DataSet data : dataSets) {
            inputs.add(data.getFeatures());
            labels.add(data.getLabels());
        }

        DataSet ds = new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])),
                        Nd4j.vstack(labels.toArray(new INDArray[0])));
        if (preProcessor != null)
            preProcessor.preProcess(ds);
        return new Tuple2<>((double) batchNumCount, ds);

    }

}
