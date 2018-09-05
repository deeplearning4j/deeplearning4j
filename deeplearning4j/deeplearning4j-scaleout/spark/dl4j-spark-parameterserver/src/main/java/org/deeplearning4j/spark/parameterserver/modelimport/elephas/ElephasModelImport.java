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

package org.deeplearning4j.spark.parameterserver.modelimport.elephas;

import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.*;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.impl.repartitioner.DefaultRepartitioner;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;

import java.io.IOException;
import java.util.Map;

/**
 * Reads HDF5-persisted Elephas models stored with `model.save()` for both underlying
 * `Sequential` and `Model` Keras models
 *
 * @author Max Pumperla
 *
 */
public class ElephasModelImport {

    private static final String DISTRIBUTED_CONFIG = "distributed_config";
    private static final RDDTrainingApproach APPROACH = RDDTrainingApproach.Export;
    private static final int WORKERS_PER_NODE = -1;
    private static final double UPDATES_THRESHOLD = 1e-3;

    /**
     * Load Elephas model stored using model.save(...) in case that the underlying Keras
     * model is a functional `Model` instance, which corresponds to a DL4J SparkComputationGraph.
     *
     * @param sparkContext                            Java SparkContext
     * @param modelHdf5Filename                       Path to HDF5 archive storing Elephas Model
     * @return SparkComputationGraph                  Spark computation graph
     *
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see SparkComputationGraph
     */
    public static SparkComputationGraph importElephasModelAndWeights(JavaSparkContext sparkContext,
                                                           String modelHdf5Filename)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelHdf5Filename, true);

        Map<String, Object> distributedProperties = distributedTrainingMap(modelHdf5Filename);
        TrainingMaster tm = getTrainingMaster(distributedProperties);

        return new SparkComputationGraph(sparkContext, model, tm);
    }

    /**
     * Load Elephas model stored using model.save(...) in case that the underlying Keras
     * model is a functional `Sequential` instance, which corresponds to a DL4J SparkDl4jMultiLayer.
     *
     * @param sparkContext                            Java SparkContext
     * @param modelHdf5Filename                       Path to HDF5 archive storing Elephas model
     * @return SparkDl4jMultiLayer                    Spark computation graph
     *
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see SparkDl4jMultiLayer
     */
    public static SparkDl4jMultiLayer importElephasSequentialModelAndWeights(JavaSparkContext sparkContext,
                                                                     String modelHdf5Filename)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(
                modelHdf5Filename, true);

        Map<String, Object> distributedProperties = distributedTrainingMap(modelHdf5Filename);
        TrainingMaster tm = getTrainingMaster(distributedProperties);

        return new SparkDl4jMultiLayer(sparkContext, model, tm);
    }

    private static Map<String, Object> distributedTrainingMap(String modelHdf5Filename)
            throws UnsupportedKerasConfigurationException, IOException {
        Hdf5Archive archive = new Hdf5Archive(modelHdf5Filename);
        String initialModelJson = archive.readAttributeAsJson(DISTRIBUTED_CONFIG);
        return KerasModelUtils.parseJsonString(initialModelJson);
    }

    private static TrainingMaster getTrainingMaster(Map<String, Object> distributedProperties) {
        Map innerConfig = (Map) distributedProperties.get("config");

        Integer numWorkers = (Integer) innerConfig.get("num_workers");
        int batchSize = (int) innerConfig.get("batch_size");
        String mode = (String) innerConfig.get("mode");
        int rddDataSetNumExamples = batchSize;

        TrainingMaster tm;
        if (mode.equals("synchronous")) {
            tm = new ParameterAveragingTrainingMaster.Builder(numWorkers, rddDataSetNumExamples)
                    .rddTrainingApproach(APPROACH)
                    .batchSizePerWorker(batchSize)
                    .aggregationDepth(2) // we leave this as default
                    .averagingFrequency(1) // TODO in number of batches
                    .workerPrefetchNumBatches(0) // default, no pre-fetching
                    .repartionData(Repartition.Always)
                    .repartitionStrategy(RepartitionStrategy.Balanced)
                    .saveUpdater(false)
                    .collectTrainingStats(false)
                    .build();
        } else {
            VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                    .build();
            tm = new SharedTrainingMaster.Builder(voidConfiguration, rddDataSetNumExamples)
                    .shakeFrequency(0) // TODO disabled by default
                    .minUpdatesThreshold(1e-5) // TODO
                    .updatesThreshold(UPDATES_THRESHOLD)
                    .rddTrainingApproach(APPROACH)
                    .batchSizePerWorker(batchSize)
                    .workersPerNode(WORKERS_PER_NODE) // TODO
                    .rddTrainingApproach(RDDTrainingApproach.Export)
                    .repartitioner(new DefaultRepartitioner())
                    .collectTrainingStats(false)
                    .stepDelay(50) // TODO
                    .stepTrigger(0.05) // TODO
                    .workerPrefetchNumBatches(0) // default, no pre-fetching
                    .thresholdStep(1e-5) // TODO
                    .transport(new RoutedTransport())
                    .build();
        }
        return tm;
    }
}
