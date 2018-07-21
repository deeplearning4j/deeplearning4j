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

package org.deeplearning4j.spark.parameterserver.training;

import lombok.Getter;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingHook;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.worker.NetBroadcastTuple;
import org.deeplearning4j.spark.impl.paramavg.BaseTrainingWorker;
import org.deeplearning4j.spark.parameterserver.conf.SharedTrainingConfiguration;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.primitives.Pair;

/**
 * @author raver119@gmail.com
 */
public class SharedTrainingWorker extends BaseTrainingWorker<SharedTrainingResult>
                implements TrainingWorker<SharedTrainingResult> {

    @Getter
    private final long instanceId;
    @Getter
    private final Broadcast<NetBroadcastTuple> broadcastModel;
    @Getter
    private final Broadcast<SharedTrainingConfiguration> broadcastConfiguration;

    public SharedTrainingWorker(long instanceId, Broadcast<NetBroadcastTuple> broadcastModel,
                    Broadcast<SharedTrainingConfiguration> broadcastConfiguration) {
        this.instanceId = instanceId;
        // our initial model is stored here.
        this.broadcastModel = broadcastModel;
        this.broadcastConfiguration = broadcastConfiguration;
    }

    @Override
    public void removeHook(TrainingHook trainingHook) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void addHook(TrainingHook trainingHook) {
        throw new UnsupportedOperationException();
    }

    @Override
    public MultiLayerNetwork getInitialModel() {
        // This method will be called ONLY once, in master thread
        NetBroadcastTuple tuple = broadcastModel.getValue();
        if (tuple.getConfiguration() != null) {
            MultiLayerConfiguration conf = tuple.getConfiguration();
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();

            if (tuple.getParameters() != null)
                network.setParams(tuple.getParameters());

            // we can assign properly, without
            if (tuple.getUpdaterState() != null)
                network.getUpdater().getStateViewArray().assign(tuple.getUpdaterState());

            return network;
        } else
            return null;
    }

    @Override
    public ComputationGraph getInitialModelGraph() {
        // This method will be called ONLY once, in master thread
        NetBroadcastTuple tuple = broadcastModel.getValue();
        if (tuple.getGraphConfiguration() != null) {
            ComputationGraphConfiguration conf = tuple.getGraphConfiguration();
            ComputationGraph network = new ComputationGraph(conf);
            network.init();

            if (tuple.getParameters() != null)
                network.setParams(tuple.getParameters());

            if (tuple.getUpdaterState() != null)
                network.getUpdater().getUpdaterStateViewArray().assign(tuple.getUpdaterState());

            return network;
        } else
            return null;
    }

    @Override
    public SharedTrainingResult processMinibatch(DataSet dataSet, MultiLayerNetwork network, boolean isLast) {
        /*
            We're not really going to use this method for training.
            Partitions will be mapped to ParallelWorker threads dynamically, wrt thread/device affinity.
            So plan is simple: we're going to use individual partitions to feed main worker
         */
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult processMinibatch(DataSet dataSet, ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult processMinibatch(MultiDataSet dataSet, ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet,
                    MultiLayerNetwork network, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(DataSet dataSet,
                    ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> processMinibatchWithStats(MultiDataSet dataSet,
                    ComputationGraph graph, boolean isLast) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult getFinalResult(MultiLayerNetwork network) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult getFinalResult(ComputationGraph network) {
        throw new UnsupportedOperationException();
    }

    @Override
    public SharedTrainingResult getFinalResultNoData() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> getFinalResultNoDataWithStats() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> getFinalResultWithStats(MultiLayerNetwork network) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<SharedTrainingResult, SparkTrainingStats> getFinalResultWithStats(ComputationGraph graph) {
        throw new UnsupportedOperationException();
    }

    @Override
    public WorkerConfiguration getDataConfiguration() {
        throw new UnsupportedOperationException();
    }
}
