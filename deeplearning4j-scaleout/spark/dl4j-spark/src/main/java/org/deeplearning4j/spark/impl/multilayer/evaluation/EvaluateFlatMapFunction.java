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

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**Function to evaluate data (classification), in a distributed manner
 * Flat map function used to batch examples for computational efficiency + reduce number of Evaluation objects returned
 * for network efficiency.
 * @author Alex Black
 */
public class EvaluateFlatMapFunction implements FlatMapFunction<Iterator<DataSet>, Evaluation> {

    protected static Logger log = LoggerFactory.getLogger(EvaluateFlatMapFunction.class);

    protected Broadcast<String> json;
    protected Broadcast<INDArray> params;
    protected Broadcast<List<String>> labels;
    protected int evalBatchSize;

    /**
     * @param json Network configuration (json format)
     * @param params Network parameters
     * @param evalBatchSize Max examples per evaluation. Do multiple separate forward passes if data exceeds
     *                              this. Used to avoid doing too many at once (and hence memory issues)
     * @param labels list of string labels
     */
    public EvaluateFlatMapFunction(Broadcast<String> json, Broadcast<INDArray> params, int evalBatchSize,
                                   Broadcast<List<String>> labels){
        this.json = json;
        this.params = params;
        this.evalBatchSize = evalBatchSize;
        this.labels = labels;
    }

    @Override
    public Iterable<Evaluation> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.emptyList();
        }

        MultiLayerNetwork network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json.getValue()));
        network.init();
        INDArray val = params.value();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParameters(val);

        Evaluation evaluation;
        if(labels != null) evaluation = new Evaluation(labels.getValue());
        else evaluation = new Evaluation();

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
                    evaluation.evalTimeSeries(data.getLabels(),out);
                } else {
                    evaluation.evalTimeSeries(data.getLabels(),out,data.getLabelsMaskArray());
                }
            } else {
                evaluation.eval(data.getLabels(),out);
            }
        }

        if (log.isDebugEnabled()) {
            log.debug("Evaluated {} examples ", totalCount);
        }

        return Collections.singletonList(evaluation);
    }
}
