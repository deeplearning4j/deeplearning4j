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

package org.deeplearning4j.spark.impl.graph.evaluation;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.broadcast.Broadcast;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.api.loader.MultiDataSetLoader;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.datasets.iterator.loader.DataSetLoaderIterator;
import org.deeplearning4j.datasets.iterator.loader.MultiDataSetLoaderIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.data.loader.RemoteFileSourceFactory;
import org.deeplearning4j.spark.impl.evaluation.EvaluationRunner;
import org.deeplearning4j.spark.iterator.SparkAMDSI;
import org.nd4j.api.loader.SourceFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Collections;
import java.util.Iterator;
import java.util.concurrent.Future;

/**
 * Function to evaluate data (using one or more IEvaluation instances), in a distributed manner
 * Flat map function used to batch examples for computational efficiency + reduce number of IEvaluation objects returned
 * for network efficiency.
 *
 * @author Alex Black
 */
public class IEvaluateMDSPathsFlatMapFunction
                extends BaseFlatMapFunctionAdaptee<Iterator<String>, IEvaluation[]> {

    public IEvaluateMDSPathsFlatMapFunction(Broadcast<String> json, Broadcast<INDArray> params, int evalNumWorkers, int evalBatchSize,
                                            DataSetLoader dsLoader, MultiDataSetLoader mdsLoader, IEvaluation... evaluations) {
        super(new IEvaluateMDSPathsFlatMapFunctionAdapter(json, params, evalNumWorkers, evalBatchSize, dsLoader, mdsLoader, evaluations));
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
class IEvaluateMDSPathsFlatMapFunctionAdapter implements FlatMapFunctionAdapter<Iterator<String>, IEvaluation[]> {

    protected Broadcast<String> json;
    protected Broadcast<INDArray> params;
    protected int evalNumWorkers;
    protected int evalBatchSize;
    protected DataSetLoader dsLoader;
    protected MultiDataSetLoader mdsLoader;
    protected IEvaluation[] evaluations;

    /**
     * @param json Network configuration (json format)
     * @param params Network parameters
     * @param evalBatchSize Max examples per evaluation. Do multiple separate forward passes if data exceeds
     *                              this. Used to avoid doing too many at once (and hence memory issues)
     * @param evaluations Initial evaulation instance (i.e., empty Evaluation or RegressionEvaluation instance)
     */
    public IEvaluateMDSPathsFlatMapFunctionAdapter(Broadcast<String> json, Broadcast<INDArray> params, int evalNumWorkers, int evalBatchSize,
                                                   DataSetLoader dsLoader, MultiDataSetLoader mdsLoader, IEvaluation[] evaluations) {
        this.json = json;
        this.params = params;
        this.evalNumWorkers = evalNumWorkers;
        this.evalBatchSize = evalBatchSize;
        this.dsLoader = dsLoader;
        this.mdsLoader = mdsLoader;
        this.evaluations = evaluations;
    }

    @Override
    public Iterable<IEvaluation[]> call(Iterator<String> paths) throws Exception {
        if (!paths.hasNext()) {
            return Collections.emptyList();
        }

        MultiDataSetIterator iter;
        if(dsLoader != null){
            DataSetIterator dsIter = new DataSetLoaderIterator(paths, dsLoader, new RemoteFileSourceFactory());
            iter = new MultiDataSetIteratorAdapter(dsIter);
        } else {
            iter = new MultiDataSetLoaderIterator(paths, mdsLoader, new RemoteFileSourceFactory());
        }

        Future<IEvaluation[]> f = EvaluationRunner.getInstance().execute(evaluations, evalNumWorkers, evalBatchSize, null, iter, true, json, params);
        IEvaluation[] result = f.get();
        if(result == null){
            return Collections.emptyList();
        } else {
            return Collections.singletonList(result);
        }
    }
}
