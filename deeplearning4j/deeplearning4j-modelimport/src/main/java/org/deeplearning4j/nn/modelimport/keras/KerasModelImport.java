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

package org.deeplearning4j.nn.modelimport.keras;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.io.InputStream;

/**
 * Reads stored Keras configurations and weights from one of two archives:
 * either as
 *
 * - a single HDF5 file storing model and training JSON configurations and weights
 * - separate text file storing model JSON configuration and HDF5 file storing weights.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class KerasModelImport {
    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Stream       InputStream containing HDF5 archive storing Keras Model
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return ComputationGraph
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(
            InputStream modelHdf5Stream,
            boolean enforceTrainingConfig) {
        throw new UnsupportedOperationException("Reading HDF5 files from InputStreams currently unsupported.");
    }

    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Stream InputStream containing HDF5 archive storing Keras Model
     * @return ComputationGraph
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(InputStream modelHdf5Stream) {
        throw new UnsupportedOperationException("Reading HDF5 files from InputStreams currently unsupported.");
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Stream       InputStream containing HDF5 archive storing Keras Sequential model
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return ComputationGraph
     * @see ComputationGraph
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(InputStream modelHdf5Stream,
                                                                         boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        throw new UnsupportedOperationException("Reading HDF5 files from InputStreams currently unsupported.");
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Stream InputStream containing HDF5 archive storing Keras Sequential model
     * @return ComputationGraph
     * @see ComputationGraph
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(InputStream modelHdf5Stream)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        throw new UnsupportedOperationException("Reading HDF5 files from InputStreams currently unsupported.");
    }

    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Filename     path to HDF5 archive storing Keras Model
     * @param inputShape            optional input shape for models that come without such (e.g. notop = false models)
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return ComputationGraph
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelHdf5Filename, int[] inputShape,
                                                              boolean enforceTrainingConfig)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder.modelHdf5Filename(modelHdf5Filename)
                .enforceTrainingConfig(enforceTrainingConfig).inputShape(inputShape).buildModel();
        return kerasModel.getComputationGraph();
    }


    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Filename     path to HDF5 archive storing Keras Model
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return ComputationGraph
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelHdf5Filename, boolean enforceTrainingConfig)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder.modelHdf5Filename(modelHdf5Filename)
                .enforceTrainingConfig(enforceTrainingConfig).buildModel();
        return kerasModel.getComputationGraph();
    }

    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Filename path to HDF5 archive storing Keras Model
     * @return ComputationGraph
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelHdf5Filename)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder().modelHdf5Filename(modelHdf5Filename)
                .enforceTrainingConfig(true).buildModel();
        return kerasModel.getComputationGraph();
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Filename     path to HDF5 archive storing Keras Sequential model
     * @param inputShape            optional input shape for models that come without such (e.g. notop = false models)
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelHdf5Filename,
                                                                         int[] inputShape,
                                                                         boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelHdf5Filename)
                .enforceTrainingConfig(enforceTrainingConfig).inputShape(inputShape).buildSequential();
        return kerasModel.getMultiLayerNetwork();
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Filename     path to HDF5 archive storing Keras Sequential model
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelHdf5Filename,
                                                                         boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelHdf5Filename)
                .enforceTrainingConfig(enforceTrainingConfig).buildSequential();
        return kerasModel.getMultiLayerNetwork();
    }

    /**
     * Load Keras Sequential model saved using model.save_model(...).
     *
     * @param modelHdf5Filename path to HDF5 archive storing Keras Sequential model
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelHdf5Filename)
                .enforceTrainingConfig(true).buildSequential();
        return kerasModel.getMultiLayerNetwork();
    }

    /**
     * Load Keras (Functional API) Model for which the configuration and weights were
     * saved separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename     path to JSON file storing Keras Model configuration
     * @param weightsHdf5Filename   path to HDF5 archive storing Keras model weights
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return ComputationGraph
     * @throws IOException IO exception
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelJsonFilename, String weightsHdf5Filename,
                                                              boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .enforceTrainingConfig(false)
                .weightsHdf5FilenameNoRoot(weightsHdf5Filename).enforceTrainingConfig(enforceTrainingConfig)
                .buildModel();
        return kerasModel.getComputationGraph();
    }

    /**
     * Load Keras (Functional API) Model for which the configuration and weights were
     * saved separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename   path to JSON file storing Keras Model configuration
     * @param weightsHdf5Filename path to HDF5 archive storing Keras model weights
     * @return ComputationGraph
     * @throws IOException IO exception
     * @see ComputationGraph
     */
    public static ComputationGraph importKerasModelAndWeights(String modelJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .enforceTrainingConfig(false)
                .weightsHdf5FilenameNoRoot(weightsHdf5Filename).enforceTrainingConfig(true).buildModel();
        return kerasModel.getComputationGraph();
    }

    /**
     * Load Keras Sequential model for which the configuration and weights were
     * saved separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename     path to JSON file storing Keras Sequential model configuration
     * @param weightsHdf5Filename   path to HDF5 archive storing Keras model weights
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelJsonFilename,
                                                                         String weightsHdf5Filename,
                                                                         boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .weightsHdf5FilenameNoRoot(weightsHdf5Filename).enforceTrainingConfig(enforceTrainingConfig)
                .buildSequential();
        return kerasModel.getMultiLayerNetwork();
    }

    /**
     * Load Keras Sequential model for which the configuration and weights were
     * saved separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename   path to JSON file storing Keras Sequential model configuration
     * @param weightsHdf5Filename path to HDF5 archive storing Keras model weights
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerNetwork importKerasSequentialModelAndWeights(String modelJsonFilename,
                                                                         String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .weightsHdf5FilenameNoRoot(weightsHdf5Filename).enforceTrainingConfig(false).buildSequential();
        return kerasModel.getMultiLayerNetwork();
    }

    /**
     * Load Keras (Functional API) Model for which the configuration was saved
     * separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename     path to JSON file storing Keras Model configuration
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return ComputationGraph
     * @throws IOException IO exception
     * @see ComputationGraph
     */
    public static ComputationGraphConfiguration importKerasModelConfiguration(String modelJsonFilename,
                                                                              boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .enforceTrainingConfig(enforceTrainingConfig).buildModel();
        return kerasModel.getComputationGraphConfiguration();
    }

    /**
     * Load Keras (Functional API) Model for which the configuration was saved
     * separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename path to JSON file storing Keras Model configuration
     * @return ComputationGraph
     * @throws IOException IO exception
     * @see ComputationGraph
     */
    public static ComputationGraphConfiguration importKerasModelConfiguration(String modelJsonFilename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModel kerasModel = new KerasModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .enforceTrainingConfig(false).buildModel();
        return kerasModel.getComputationGraphConfiguration();
    }

    /**
     * Load Keras Sequential model for which the configuration was saved
     * separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename     path to JSON file storing Keras Sequential model configuration
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerConfiguration importKerasSequentialConfiguration(String modelJsonFilename,
                                                                             boolean enforceTrainingConfig)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .enforceTrainingConfig(enforceTrainingConfig).buildSequential();
        return kerasModel.getMultiLayerConfiguration();
    }

    /**
     * Load Keras Sequential model for which the configuration was saved
     * separately using calls to model.to_json() and model.save_weights(...).
     *
     * @param modelJsonFilename path to JSON file storing Keras Sequential model configuration
     * @return MultiLayerNetwork
     * @throws IOException IO exception
     * @see MultiLayerNetwork
     */
    public static MultiLayerConfiguration importKerasSequentialConfiguration(String modelJsonFilename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel = new KerasSequentialModel().modelBuilder().modelJsonFilename(modelJsonFilename)
                .enforceTrainingConfig(false).buildSequential();
        return kerasModel.getMultiLayerConfiguration();
    }
}
