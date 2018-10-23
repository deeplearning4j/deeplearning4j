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
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.impl.repartitioner.DefaultRepartitioner;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

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

    private static TrainingMaster getTrainingMaster(Map<String, Object> distributedProperties)
            throws InvalidKerasConfigurationException {
        Map innerConfig = (Map) distributedProperties.get("config");

        Integer numWorkers = (Integer) innerConfig.get("num_workers");
        int batchSize = (int) innerConfig.get("batch_size");

        String mode = "synchronous";
        if (innerConfig.containsKey("mode")) {
            mode = (String) innerConfig.get("mode");
        } else {
            throw new InvalidKerasConfigurationException("Couldn't find mode field.");
        }

        // TODO: Create InvalidElephasConfigurationException
        boolean collectStats = false;
        if (innerConfig.containsKey("collect_stats"))
            collectStats = (boolean) innerConfig.get("collect_stats");

        int numBatchesPrefetch = 0;
        if (innerConfig.containsKey("num_batches_prefetch"))
            numBatchesPrefetch = (int) innerConfig.get("num_batches_prefetch");


    TrainingMaster tm;
        if (mode.equals("synchronous")) {
            int averagingFrequency = 5;
            if (innerConfig.containsKey("averaging_frequency"))
                averagingFrequency = (int) innerConfig.get("averaging_frequency");

            tm = new ParameterAveragingTrainingMaster.Builder(numWorkers, batchSize)
                    .collectTrainingStats(collectStats)
                    .batchSizePerWorker(batchSize)
                    .averagingFrequency(averagingFrequency)
                    .workerPrefetchNumBatches(numBatchesPrefetch)
                    .aggregationDepth(2) // we leave this as default
                    .repartionData(Repartition.Always)
                    .rddTrainingApproach(APPROACH)
                    .repartitionStrategy(RepartitionStrategy.Balanced)
                    .saveUpdater(false)
                    .build();
        } else if (mode.equals("asynchronous")){
            int shakeFrequency = 0;
            if (innerConfig.containsKey("shake_frequency"))
                shakeFrequency = (int) innerConfig.get("shake_frequency");

            double minThreshold = 1e-5;
            if (innerConfig.containsKey("min_threshold"))
                minThreshold = (double) innerConfig.get("min_threshold");

            double updateThreshold = 1e-3;
            if (innerConfig.containsKey("update_threshold"))
                minThreshold = (double) innerConfig.get("update_threshold");

            int workersPerNode = -1;
            if (innerConfig.containsKey("workers_per_node"))
                workersPerNode = (int) innerConfig.get("workers_per_node");

            int stepDelay = 50;
            if (innerConfig.containsKey("step_delay"))
                stepDelay = (int) innerConfig.get("step_delay");

            double stepTrigger = 0.05;
            if (innerConfig.containsKey("step_trigger"))
                stepTrigger = (double) innerConfig.get("step_trigger");

            double thresholdStep = 1e-5;
            if (innerConfig.containsKey("threshold_step"))
                thresholdStep = (double) innerConfig.get("threshold_step");


            VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                    .build();
            tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSize)
                    .shakeFrequency(shakeFrequency)
                    .minUpdatesThreshold(minThreshold)
                    .updatesThreshold(updateThreshold)
                    .batchSizePerWorker(batchSize)
                    .workersPerNode(workersPerNode)
                    .collectTrainingStats(collectStats)
                    .stepDelay(stepDelay)
                    .stepTrigger(stepTrigger)
                    .workerPrefetchNumBatches(numBatchesPrefetch)
                    .thresholdStep(thresholdStep)
                    .rddTrainingApproach(APPROACH)
                    .repartitioner(new DefaultRepartitioner())
                    .build();
        } else {
            throw new InvalidKerasConfigurationException("Unknown mode " + mode);
        }
        return tm;
    }
}
