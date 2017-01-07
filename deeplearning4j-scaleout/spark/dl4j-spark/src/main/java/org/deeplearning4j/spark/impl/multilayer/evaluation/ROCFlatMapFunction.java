/*
 *
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
 *
 */

package org.deeplearning4j.spark.impl.multilayer.evaluation;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Function to perform ROC evaluation of a classifier (classification), in a distributed manner
 * Flat map function used to batch examples for computational efficiency + reduce number of ROC objects returned
 * for efficiency.
 *
 * @author Alex Black
 */
@Slf4j
public class ROCFlatMapFunction implements FlatMapFunction<Iterator<DataSet>, ROC> {
    private Broadcast<String> json;
    private Broadcast<INDArray> params;
    private int numThresholdSteps;
    private int evalBatchSize;

    /**
     * @param json Network configuration (json format)
     * @param params Network parameters
     * @param evalBatchSize Max examples per evaluation. Do multiple separate forward passes if data exceeds
     *                              this. Used to avoid doing too many at once (and hence memory issues)
     */
    public ROCFlatMapFunction(Broadcast<String> json, Broadcast<INDArray> params, int numThresholdSteps,
                              int evalBatchSize){
        this.json = json;
        this.params = params;
        this.numThresholdSteps = numThresholdSteps;
        this.evalBatchSize = evalBatchSize;
    }

    @Override
    public Iterable<ROC> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.emptyList();
        }

        MultiLayerNetwork network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json.getValue()));
        network.init();
        INDArray val = params.value().unsafeDuplication();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParameters(val);

        ROC roc = new ROC(numThresholdSteps);

        List<DataSet> collect = new ArrayList<>();
        int totalCount = 0;
        while (dataSetIterator.hasNext()) {
            collect.clear();
            int nExamples = 0;
            while (dataSetIterator.hasNext() && nExamples < evalBatchSize) {
                DataSet next = dataSetIterator.next();
                nExamples += next.numExamples();
                collect.add(next);
            }
            totalCount += nExamples;

            DataSet data = DataSet.merge(collect, false);


            INDArray out;
            if(data.hasMaskArrays()) {
                out = network.output(data.getFeatureMatrix(), false, data.getFeaturesMaskArray(), data.getLabelsMaskArray());
            } else {
                out = network.output(data.getFeatureMatrix(), false);
            }


            if(data.getLabels().rank() == 3){
                if(data.getLabelsMaskArray() == null){
                    roc.evalTimeSeries(data.getLabels(),out);
                } else {
                    roc.evalTimeSeries(data.getLabels(),out,data.getLabelsMaskArray());
                }
            } else {
                roc.eval(data.getLabels(),out);
            }
        }

        if (log.isDebugEnabled()) {
            log.debug("ROC: Evaluated {} examples ", totalCount);
        }

        return Collections.singletonList(roc);
    }
}
