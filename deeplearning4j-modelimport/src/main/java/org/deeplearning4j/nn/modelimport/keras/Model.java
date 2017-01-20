/*
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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.io.InputStream;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;

/**
 * Routines for importing saved Keras models.
 *
 * @author dave@skymind.io
 *
 * @deprecated Use {@link KerasModelImport} instead.
 */
@Deprecated
@Slf4j
public class Model {

    private Model() {}

    /**
     * Load Keras model saved using model.save_model(...).
     *
     * @param  modelHdf5Stream      input stream storing Keras Sequential model
     * @return                      DL4J MultiLayerNetwork
     * @see    MultiLayerNetwork
     * @throws UnsupportedKerasConfigurationException
     * @throws IOException
     * @throws org.deeplearning4j.nn.api.Model
     * @deprecated Use {@link KerasModelImport#importKerasModelAndWeights} instead
     */
    public static org.deeplearning4j.nn.api.Model importModelInputStream(InputStream modelHdf5Stream)
            throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        return KerasModelImport.importKerasModelAndWeights(modelHdf5Stream, false);
    }

    /**
     * Imports a Keras Sequential model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param  modelHdf5Stream input stream storing Keras Sequential model
     * @return                   DL4J MultiLayerNetwork
     * @see    MultiLayerNetwork
     * @throws UnsupportedKerasConfigurationException
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @deprecated Use {@link KerasModelImport#importKerasSequentialModelAndWeights} instead
     */
    public static MultiLayerNetwork importSequentialModelInputStream(InputStream modelHdf5Stream)
            throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        return KerasModelImport.importKerasSequentialModelAndWeights(modelHdf5Stream, false);
    }

    /**
     * Imports a Keras Functional API model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param modelHdf5Stream  input stream storing storing Keras Functional API model
     * @return                   DL4J ComputationGraph
     * @throws UnsupportedKerasConfigurationException
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @deprecated Use {@link KerasModelImport#importKerasModelAndWeights} instead
     */
    public static ComputationGraph importFunctionalApiModelInputStream(InputStream modelHdf5Stream)
            throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        return KerasModelImport.importKerasModelAndWeights(modelHdf5Stream, false);
    }

    /**
     * Load Keras model saved using model.save_model(...).
     *
     * @param modelHdf5Filename    path to HDF5 archive storing Keras model
     * @return                     DL4J Model interface
     * @throws IOException
     * @see org.deeplearning4j.nn.api.Model
     * @deprecated Use {@link KerasModelImport#importKerasModelAndWeights} instead
     */
    @Deprecated
    public static org.deeplearning4j.nn.api.Model importModel(String modelHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return KerasModelImport.importKerasModelAndWeights(modelHdf5Filename, false);
    }

    /**
     * Load Keras model where the config and weights were saved separately using calls to
     * model.to_json() and model.save_weights(...).
     *
     * @param configJsonFilename    path to JSON file storing Keras Functional API model configuration
     * @param weightsHdf5Filename   path to HDF5 archive storing Keras Functional API model weights
     * @return                      DL4J Model interface
     * @throws IOException
     * @see org.deeplearning4j.nn.api.Model
     * @deprecated Use {@link KerasModelImport#importKerasModelAndWeights} instead
     */
    @Deprecated
    public static org.deeplearning4j.nn.api.Model importModel(String configJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return KerasModelImport.importKerasModelAndWeights(configJsonFilename, weightsHdf5Filename, false);
    }

    /**
     * Imports a Keras Sequential model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param modelHdf5Filename    path to HDF5 archive storing Keras Sequential model
     * @return                     DL4J MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     * @deprecated Use {@link KerasModelImport#importKerasSequentialModelAndWeights} instead
     */
    @Deprecated
    public static MultiLayerNetwork importSequentialModel(String modelHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return KerasModelImport.importKerasSequentialModelAndWeights(modelHdf5Filename, false);
    }

    /**
     * Imports a Keras Functional API model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param modelHdf5Filename  path to HDF5 archive storing Keras Functional API model
     * @return                   DL4J ComputationGraph
     * @throws IOException
     * @see    ComputationGraph
     * @deprecated Use {@link KerasModelImport#importKerasModelAndWeights} instead
     */
    @Deprecated
    public static ComputationGraph importFunctionalApiModel(String modelHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return KerasModelImport.importKerasModelAndWeights(modelHdf5Filename, false);
    }

    /**
     * Imports a Keras Sequential model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param configJsonFilename     path to JSON file storing Keras Functional API model configuration
     * @param weightsHdf5Filename    path to HDF5 archive storing Keras Functional API model weights
     * @return                       DL4J MultiLayerNetwork
     * @throws IOException
     * @see MultiLayerNetwork
     * @deprecated Use {@link KerasModelImport#importKerasSequentialModelAndWeights} instead
     */
    @Deprecated
    public static MultiLayerNetwork importSequentialModel(String configJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return KerasModelImport.importKerasSequentialModelAndWeights(configJsonFilename, weightsHdf5Filename, false);
    }

    /**
     * Imports a Keras Functional API model saved using model.save_model(...). Model
     * configuration and weights are loaded from single HDF5 archive.
     *
     * @param configJsonFilename     path to JSON file storing Keras Functional API model configuration
     * @param weightsHdf5Filename    path to HDF5 archive storing Keras Functional API model weights
     * @return                       DL4J ComputationGraph
     * @throws IOException
     * @see    ComputationGraph
     * @deprecated Use {@link KerasModelImport#importKerasModelAndWeights} instead
     */
    @Deprecated
    public static ComputationGraph importFunctionalApiModel(String configJsonFilename, String weightsHdf5Filename)
            throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return KerasModelImport.importKerasModelAndWeights(configJsonFilename, weightsHdf5Filename, false);
    }
}
