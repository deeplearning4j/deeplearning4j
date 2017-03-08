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
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Routines for importing saved Keras model configurations.
 *
 * @author dave@skymind.io
 *
 * @deprecated Use {@link KerasModelImport} instead.
 */
@Deprecated
@Slf4j
public class ModelConfiguration {

    private ModelConfiguration() {}

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param inputStream    InputStream containing Keras Sequential configuration as valid JSON.
     * @return               DL4J MultiLayerConfiguration
     * @throws IOException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    @Deprecated
    public static MultiLayerConfiguration importSequentialModelConfigFromInputStream(InputStream inputStream)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        IOUtils.copy(inputStream, byteArrayOutputStream);
        String configJson = new String(byteArrayOutputStream.toByteArray());
        return importSequentialModelConfig(configJson);
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param inputStream    InputStream containing Keras Sequential configuration as valid JSON.
     * @return               DL4J ComputationGraphConfiguration
     * @throws IOException
     */
    @Deprecated
    public static ComputationGraphConfiguration importFunctionalApiConfigFromInputStream(InputStream inputStream)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        IOUtils.copy(inputStream, byteArrayOutputStream);
        String configJson = new String(byteArrayOutputStream.toByteArray());
        return importFunctionalApiConfig(configJson);
    }

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param modelJsonFilename    Path to text file storing Keras Sequential configuration as valid JSON.
     * @return                     DL4J MultiLayerConfiguration
     * @throws IOException
     * @deprecated Use {@link KerasModelImport#importKerasSequentialConfiguration} instead
     */
    @Deprecated
    public static MultiLayerConfiguration importSequentialModelConfigFromFile(String modelJsonFilename)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return importSequentialModelConfigFromInputStream(new FileInputStream(modelJsonFilename));
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param modelJsonFilename    Path to text file storing Keras Model configuration as valid JSON.
     * @return                     DL4J ComputationGraphConfiguration
     * @throws IOException
     * @deprecated Use {@link KerasModelImport#importKerasModelConfiguration} instead
     */
    @Deprecated
    public static ComputationGraphConfiguration importFunctionalApiConfigFromFile(String modelJsonFilename)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        return importFunctionalApiConfigFromInputStream(new FileInputStream(modelJsonFilename));
    }

    /**
     * Imports a Keras Sequential model configuration saved using call to model.to_json().
     *
     * @param modelJson    String storing Keras Sequential configuration as valid JSON.
     * @return             DL4J MultiLayerConfiguration
     * @throws IOException
     * @deprecated Use {@link org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel} instead
     */
    @Deprecated
    public static MultiLayerConfiguration importSequentialModelConfig(String modelJson)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasSequentialModel kerasModel =
                        new KerasSequentialModel.ModelBuilder().modelJson(modelJson).buildSequential();
        return kerasModel.getMultiLayerConfiguration();
    }

    /**
     * Imports a Keras Functional API model configuration saved using call to model.to_json().
     *
     * @param modelJson    String storing Keras Model configuration as valid JSON.
     * @return             DL4J ComputationGraphConfiguration
     * @throws IOException
     * @deprecated Use {@link org.deeplearning4j.nn.modelimport.keras.KerasModel} instead
     */
    @Deprecated
    public static ComputationGraphConfiguration importFunctionalApiConfig(String modelJson)
                    throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        KerasModel kerasModel = new KerasModel.ModelBuilder().modelJson(modelJson).buildModel();
        return kerasModel.getComputationGraphConfiguration();
    }
}
