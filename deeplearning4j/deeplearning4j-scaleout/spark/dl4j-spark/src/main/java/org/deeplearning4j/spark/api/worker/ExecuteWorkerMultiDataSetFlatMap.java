/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.api.worker;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.nd4j.linalg.dataset.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.stats.StatsCalculationHelper;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.Collections;
import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on MultiDataSets. Used only in SparkComputationGraph implementation.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class ExecuteWorkerMultiDataSetFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<MultiDataSet>, R> {

    private final TrainingWorker<R> worker;

    @Override
    public Iterator<R> call(Iterator<MultiDataSet> dataSetIterator) throws Exception {
        WorkerConfiguration dataConfig = worker.getDataConfiguration();

        boolean stats = dataConfig.isCollectTrainingStats();
        StatsCalculationHelper s = (stats ? new StatsCalculationHelper() : null);
        if (stats)
            s.logMethodStartTime();

        if (!dataSetIterator.hasNext()) {
            if (stats)
                s.logReturnTime();
            //TODO return the results...
            return Collections.emptyIterator(); //Sometimes: no data
        }

        int batchSize = dataConfig.getBatchSizePerWorker();
        final int prefetchCount = dataConfig.getPrefetchNumBatches();

        MultiDataSetIterator batchedIterator = new IteratorMultiDataSetIterator(dataSetIterator, batchSize);
        if (prefetchCount > 0) {
            batchedIterator = new AsyncMultiDataSetIterator(batchedIterator, prefetchCount);
        }

        try {
            if (stats)
                s.logInitialModelBefore();
            ComputationGraph net = worker.getInitialModelGraph();
            if (stats)
                s.logInitialModelAfter();

            int miniBatchCount = 0;
            int maxMinibatches = (dataConfig.getMaxBatchesPerWorker() > 0 ? dataConfig.getMaxBatchesPerWorker()
                            : Integer.MAX_VALUE);

            while (batchedIterator.hasNext() && miniBatchCount++ < maxMinibatches) {
                if (stats)
                    s.logNextDataSetBefore();
                MultiDataSet next = batchedIterator.next();

                if (stats)
                    s.logNextDataSetAfter(next.getFeatures(0).size(0));

                if (stats) {
                    s.logProcessMinibatchBefore();
                    Pair<R, SparkTrainingStats> result =
                                    worker.processMinibatchWithStats(next, net, !batchedIterator.hasNext());
                    s.logProcessMinibatchAfter();
                    if (result != null) {
                        //Terminate training immediately
                        s.logReturnTime();
                        SparkTrainingStats workerStats = result.getSecond();
                        SparkTrainingStats returnStats = s.build(workerStats);
                        result.getFirst().setStats(returnStats);

                        return Collections.singletonList(result.getFirst()).iterator();
                    }
                } else {
                    R result = worker.processMinibatch(next, net, !batchedIterator.hasNext());
                    if (result != null) {
                        //Terminate training immediately
                        return Collections.singletonList(result).iterator();
                    }
                }
            }

            //For some reason, we didn't return already. Normally this shouldn't happen
            if (stats) {
                s.logReturnTime();
                Pair<R, SparkTrainingStats> pair = worker.getFinalResultWithStats(net);
                pair.getFirst().setStats(s.build(pair.getSecond()));
                return Collections.singletonList(pair.getFirst()).iterator();
            } else {
                return Collections.singletonList(worker.getFinalResult(net)).iterator();
            }
        } finally {
            Nd4j.getExecutioner().commit();

            //Make sure we shut down the async thread properly...
            if (batchedIterator instanceof AsyncMultiDataSetIterator) {
                ((AsyncMultiDataSetIterator) batchedIterator).shutdown();
            }
        }
    }
}
