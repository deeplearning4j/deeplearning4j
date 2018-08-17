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

package org.deeplearning4j.spark.parameterserver.elephas;

import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.io.IOException;

/**
 * Reads HDF5-persisted Elephas models stored with `model.save()` for both underlying
 * `Sequential` and `Model` Keras models
 *
 * @author Max Pumperla
 *
 */
public class ElephasModelImport {

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

        int rddDataSetNumExamples = 32;
        int batchSize = 32;
        double updateThreshold = 1e-3;
        int workersPerNode = -1;

        // TODO: read training config properly
        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .unicastPort(40123)
                .build();

        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, rddDataSetNumExamples)
                .updatesThreshold(updateThreshold)
                .rddTrainingApproach(APPROACH)
                .batchSizePerWorker(batchSize)
                .workersPerNode(workersPerNode)
                .build();

        return new SparkComputationGraph(sparkContext, model, tm);
    }

    /**
     * Load Elephas model stored using model.save(...) in case that the underlying Keras
     * model is a functional `Sequential` instance, which corresponds to a DL4J SparkDl4jMultiLayer.
     *
     * @param sparkContext                            Java SparkContext
     * @param modelHdf5Filename                       Path to HDF5 archive storing Elephas model
     * @return SparkComputationGraph                  Spark computation graph
     *
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see SparkComputationGraph
     */
    public static SparkDl4jMultiLayer importElephasSequentialModelAndWeights(JavaSparkContext sparkContext,
                                                                     String modelHdf5Filename)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(
                modelHdf5Filename, true);

        int rddDataSetNumExamples = 32;
        int batchSize = 32;
        double updateThreshold = 1e-3;
        int workersPerNode = -1;

        // TODO: read training config properly
        VoidConfiguration voidConfiguration = VoidConfiguration.builder().build();

        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, rddDataSetNumExamples)
                .updatesThreshold(updateThreshold)
                .rddTrainingApproach(APPROACH)
                .batchSizePerWorker(batchSize)
                .workersPerNode(workersPerNode)
                .build();

        return new SparkDl4jMultiLayer(sparkContext, model, tm);
    }
}
