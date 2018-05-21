/*-
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

package org.deeplearning4j.spark.impl.graph.evaluation;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.iterator.SparkAMDSI;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Collections;
import java.util.Iterator;

/**
 * Function to evaluate data (using one or more IEvaluation instances), in a distributed manner
 * Flat map function used to batch examples for computational efficiency + reduce number of IEvaluation objects returned
 * for network efficiency.
 *
 * @author Alex Black
 */
public class IEvaluateMDSFlatMapFunction<T extends IEvaluation>
                extends BaseFlatMapFunctionAdaptee<Iterator<MultiDataSet>, T[]> {

    public IEvaluateMDSFlatMapFunction(Broadcast<String> json, Broadcast<INDArray> params, int evalBatchSize,
                    T... evaluations) {
        super(new IEvaluateMDSFlatMapFunctionAdapter<>(json, params, evalBatchSize, evaluations));
    }
}


/**
 * Function to evaluate data (using an IEvaluation instance), in a distributed manner
 * Flat map function used to batch examples for computational efficiency + reduce number of IEvaluation objects returned
 * for network efficiency.
 *
 * @author Alex Black
 */
@Slf4j
class IEvaluateMDSFlatMapFunctionAdapter<T extends IEvaluation>
                implements FlatMapFunctionAdapter<Iterator<MultiDataSet>, T[]> {

    protected Broadcast<String> json;
    protected Broadcast<INDArray> params;
    protected int evalBatchSize;
    protected T[] evaluations;

    /**
     * @param json Network configuration (json format)
     * @param params Network parameters
     * @param evalBatchSize Max examples per evaluation. Do multiple separate forward passes if data exceeds
     *                              this. Used to avoid doing too many at once (and hence memory issues)
     * @param evaluations Initial evaulation instance (i.e., empty Evaluation or RegressionEvaluation instance)
     */
    public IEvaluateMDSFlatMapFunctionAdapter(Broadcast<String> json, Broadcast<INDArray> params, int evalBatchSize,
                    T[] evaluations) {
        this.json = json;
        this.params = params;
        this.evalBatchSize = evalBatchSize;
        this.evaluations = evaluations;
    }

    @Override
    public Iterable<T[]> call(Iterator<MultiDataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.emptyList();
        }

        INDArray val = params.value().unsafeDuplication();
        ComputationGraph graph = new ComputationGraph(ComputationGraphConfiguration.fromJson(json.getValue()));
        graph.init();
        if (val.length() != graph.numParams(false))
            throw new IllegalStateException(
                            "Network did not have same number of parameters as the broadcast set parameters");
        graph.setParams(val);

        T[] eval = graph.doEvaluation(
                        new SparkAMDSI(new IteratorMultiDataSetIterator(dataSetIterator, evalBatchSize), 2, true),
                        evaluations);
        return Collections.singletonList(eval);
    }
}
