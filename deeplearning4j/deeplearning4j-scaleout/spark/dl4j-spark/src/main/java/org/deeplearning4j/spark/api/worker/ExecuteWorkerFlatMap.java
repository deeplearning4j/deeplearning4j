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

package org.deeplearning4j.spark.api.worker;

import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.stats.StatsCalculationHelper;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on DataSets.
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
public class ExecuteWorkerFlatMap<R extends TrainingResult> extends BaseFlatMapFunctionAdaptee<Iterator<DataSet>, R> {

    public ExecuteWorkerFlatMap(TrainingWorker<R> worker) {
        super(new ExecuteWorkerFlatMapAdapter<R>(worker));
    }
}


/**
 * A FlatMapFunction for executing training on DataSets.
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
class ExecuteWorkerFlatMapAdapter<R extends TrainingResult> implements FlatMapFunctionAdapter<Iterator<DataSet>, R> {

    private final TrainingWorker<R> worker;

    public ExecuteWorkerFlatMapAdapter(TrainingWorker<R> worker) {
        this.worker = worker;
    }

    @Override
    public Iterable<R> call(Iterator<DataSet> dataSetIterator) throws Exception {
        WorkerConfiguration dataConfig = worker.getDataConfiguration();
        final boolean isGraph = dataConfig.isGraphNetwork();

        boolean stats = dataConfig.isCollectTrainingStats();
        StatsCalculationHelper s = (stats ? new StatsCalculationHelper() : null);
        if (stats)
            s.logMethodStartTime();

        if (!dataSetIterator.hasNext()) {
            if (stats) {
                s.logReturnTime();

                Pair<R, SparkTrainingStats> pair = worker.getFinalResultNoDataWithStats();
                pair.getFirst().setStats(s.build(pair.getSecond()));
                return Collections.singletonList(pair.getFirst());
            } else {
                return Collections.singletonList(worker.getFinalResultNoData());
            }
        }

        int batchSize = dataConfig.getBatchSizePerWorker();
        final int prefetchCount = dataConfig.getPrefetchNumBatches();

        DataSetIterator batchedIterator = new IteratorDataSetIterator(dataSetIterator, batchSize);
        if (prefetchCount > 0) {
            batchedIterator = new AsyncDataSetIterator(batchedIterator, prefetchCount);
        }

        try {
            MultiLayerNetwork net = null;
            ComputationGraph graph = null;
            if (stats)
                s.logInitialModelBefore();
            if (isGraph)
                graph = worker.getInitialModelGraph();
            else
                net = worker.getInitialModel();
            if (stats)
                s.logInitialModelAfter();

            int miniBatchCount = 0;
            int maxMinibatches = (dataConfig.getMaxBatchesPerWorker() > 0 ? dataConfig.getMaxBatchesPerWorker()
                            : Integer.MAX_VALUE);

            while (batchedIterator.hasNext() && miniBatchCount++ < maxMinibatches) {
                if (stats)
                    s.logNextDataSetBefore();
                DataSet next = batchedIterator.next();
                if (stats)
                    s.logNextDataSetAfter(next.numExamples());

                if (stats) {
                    s.logProcessMinibatchBefore();
                    Pair<R, SparkTrainingStats> result;
                    if (isGraph)
                        result = worker.processMinibatchWithStats(next, graph, !batchedIterator.hasNext());
                    else
                        result = worker.processMinibatchWithStats(next, net, !batchedIterator.hasNext());
                    s.logProcessMinibatchAfter();
                    if (result != null) {
                        //Terminate training immediately
                        s.logReturnTime();
                        SparkTrainingStats workerStats = result.getSecond();
                        SparkTrainingStats returnStats = s.build(workerStats);
                        result.getFirst().setStats(returnStats);

                        return Collections.singletonList(result.getFirst());
                    }
                } else {
                    R result;
                    if (isGraph)
                        result = worker.processMinibatch(next, graph, !batchedIterator.hasNext());
                    else
                        result = worker.processMinibatch(next, net, !batchedIterator.hasNext());
                    if (result != null) {
                        //Terminate training immediately
                        return Collections.singletonList(result);
                    }
                }
            }

            //For some reason, we didn't return already. Normally this shouldn't happen
            if (stats) {
                s.logReturnTime();
                Pair<R, SparkTrainingStats> pair;
                if (isGraph)
                    pair = worker.getFinalResultWithStats(graph);
                else
                    pair = worker.getFinalResultWithStats(net);
                pair.getFirst().setStats(s.build(pair.getSecond()));
                return Collections.singletonList(pair.getFirst());
            } else {
                if (isGraph)
                    return Collections.singletonList(worker.getFinalResult(graph));
                else
                    return Collections.singletonList(worker.getFinalResult(net));
            }
        } finally {
            //Make sure we shut down the async thread properly...
            Nd4j.getExecutioner().commit();

            if (batchedIterator instanceof AsyncDataSetIterator) {
                ((AsyncDataSetIterator) batchedIterator).shutdown();
            }
        }
    }
}
