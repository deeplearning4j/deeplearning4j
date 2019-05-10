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

package org.deeplearning4j.nn.modelimport.keras.layers.core;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.PermutePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasPermuteTest {

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();


    @Test
    public void testPermuteLayer() throws Exception {
        buildPermuteLayer(conf1, keras1);
        buildPermuteLayer(conf2, keras2);
    }


    private void buildPermuteLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        int[] permuteIndices = new int[]{2, 1};
        List<Integer> permuteList = new ArrayList<>();
        permuteList.add(permuteIndices[0]);
        permuteList.add(permuteIndices[1]);
        PermutePreprocessor preProcessor = getPermutePreProcessor(conf, kerasVersion, permuteList);
        assertEquals(preProcessor.getPermutationIndices()[0], permuteIndices[0]);
        assertEquals(preProcessor.getPermutationIndices()[1], permuteIndices[1]);
    }

    private PermutePreprocessor getPermutePreProcessor(KerasLayerConfiguration conf, Integer kerasVersion,
                                                       List<Integer> permuteList)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_RESHAPE());
        Map<String, Object> config = new HashMap<>();
        config.put("dims", permuteList);
        config.put(conf.getLAYER_FIELD_NAME(), "permute");
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        InputType inputType = InputType.InputTypeFeedForward.recurrent(20, 10);
        return (PermutePreprocessor) new KerasPermute(layerConfig).getInputPreprocessor(inputType);

    }
}
