/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
package org.deeplearning4j.nn.modelimport.keras.configurations;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

/**
 * Test import of Keras models.
 */
@Slf4j
public class KerasModelImportTest {

    ClassLoader classLoader = KerasModelImportTest.class.getClassLoader();


    @Test
    public void testH5WithoutTensorflowScope() throws Exception {
        MultiLayerNetwork model = loadModel("modelimport/keras/tfscope/model.h5");
        assert (model != null);
    }

    @Test
    public void testH5WithTensorflowScope() throws Exception {
        MultiLayerNetwork model = loadModel("modelimport/keras/tfscope/model.h5.with.tensorflow.scope");
        assert (model != null);
    }

    @Test
    public void testWeightAndJsonWithoutTensorflowScope() throws Exception {
        MultiLayerNetwork model = loadModel("modelimport/keras/tfscope/model.json",
                "modelimport/keras/tfscope/model.weight");
        assert (model != null);
    }

    @Test
    public void testWeightAndJsonWithTensorflowScope() throws Exception {
        MultiLayerNetwork model = loadModel(
                "modelimport/keras/tfscope/model.json.with.tensorflow.scope",
                "modelimport/keras/tfscope/model.weight.with.tensorflow.scope");
        assert (model != null);
    }

    private MultiLayerNetwork loadModel(String modelJsonFilename, String modelWeightFilename)
            throws NullPointerException {
        ClassPathResource modelResource = new ClassPathResource(modelJsonFilename, classLoader);
        ClassPathResource weightResource = new ClassPathResource(modelWeightFilename, classLoader);


        MultiLayerNetwork network = null;
        try {
            network = KerasModelImport.importKerasSequentialModelAndWeights(modelResource.getFile().getAbsolutePath(),
                    weightResource.getFile().getAbsolutePath(), false);
        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        }

        return network;
    }

    private MultiLayerNetwork loadModel(String modelFilename) {
        ClassPathResource modelResource = new ClassPathResource(modelFilename, classLoader);

        MultiLayerNetwork model = null;
        try {
            model = KerasModelImport.importKerasSequentialModelAndWeights(modelResource.getFile().getAbsolutePath());
        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        }

        return model;
    }


}
