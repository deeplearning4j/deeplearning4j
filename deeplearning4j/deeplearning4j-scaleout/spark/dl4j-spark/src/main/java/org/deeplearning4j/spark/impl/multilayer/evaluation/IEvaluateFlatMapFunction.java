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

package org.deeplearning4j.spark.impl.multilayer.evaluation;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.iterator.SparkADSI;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Collections;
import java.util.Iterator;

/**
 * Function to evaluate data (using an IEvaluation instance), in a distributed manner
 * Flat map function used to batch examples for computational efficiency + reduce number of IEvaluation objects returned
 * for network efficiency.
 *
 * @author Alex Black
 */
public class IEvaluateFlatMapFunction<T extends IEvaluation>
                extends BaseFlatMapFunctionAdaptee<Iterator<DataSet>, T[]> {

    public IEvaluateFlatMapFunction(boolean isCompGraph, Broadcast<String> json, Broadcast<INDArray> params,
                    int evalBatchSize, T... evaluations) {
        super(new IEvaluateFlatMapFunctionAdapter<>(isCompGraph, json, params, evalBatchSize, evaluations));
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
class IEvaluateFlatMapFunctionAdapter<T extends IEvaluation> implements FlatMapFunctionAdapter<Iterator<DataSet>, T[]> {

    protected boolean isCompGraph;
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
    public IEvaluateFlatMapFunctionAdapter(boolean isCompGraph, Broadcast<String> json, Broadcast<INDArray> params,
                    int evalBatchSize, T[] evaluations) {
        this.isCompGraph = isCompGraph;
        this.json = json;
        this.params = params;
        this.evalBatchSize = evalBatchSize;
        this.evaluations = evaluations;
    }

    @Override
    public Iterable<T[]> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.emptyList();
        }

        MultiLayerNetwork network = null;
        ComputationGraph graph = null;
        INDArray val = params.value().unsafeDuplication();
        if (isCompGraph) {
            graph = new ComputationGraph(ComputationGraphConfiguration.fromJson(json.getValue()));
            graph.init();
            if (val.length() != graph.numParams(false))
                throw new IllegalStateException(
                                "Network did not have same number of parameters as the broadcast set parameters");
            graph.setParams(val);

            T[] eval = graph.doEvaluation(
                            new SparkADSI(new IteratorDataSetIterator(dataSetIterator, evalBatchSize), 2, true),
                            evaluations);
            return Collections.singletonList(eval);

        } else {
            network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json.getValue()));
            network.init();
            if (val.length() != network.numParams(false))
                throw new IllegalStateException(
                                "Network did not have same number of parameters as the broadcast set parameters");
            network.setParameters(val);

            T[] eval = network.doEvaluation(
                            new SparkADSI(new IteratorDataSetIterator(dataSetIterator, evalBatchSize), 2, true),
                            evaluations);
            return Collections.singletonList(eval);
        }
    }
}
