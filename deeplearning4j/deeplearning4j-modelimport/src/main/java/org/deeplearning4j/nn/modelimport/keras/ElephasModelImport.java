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

package org.deeplearning4j.nn.modelimport.keras;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.io.IOException;

/**
 * Reads HDF5-persisted Elephas models stored with `model.save()`.
 *
 * @author Max Pumperla
 *
 * TODO: add spark dependency? or move this into spark module?
 */
public class ElephasModelImport {

    /**
     * Load Keras (Functional API) Model saved using model.save_model(...).
     *
     * @param modelHdf5Filename     path to HDF5 archive storing Keras Model
     * @param enforceTrainingConfig whether to enforce training configuration options
     * @return SparkComputationGraph Spark computation graph
     * @throws IOException                            IO exception
     * @throws InvalidKerasConfigurationException     Invalid Keras config
     * @throws UnsupportedKerasConfigurationException Unsupported Keras config
     * @see SparkComputationGraph
     */
    public static SparkComputationGraph importElephasModel(String modelHdf5Filename, boolean enforceTrainingConfig)
            throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelHdf5Filename, enforceTrainingConfig);

    }
}
