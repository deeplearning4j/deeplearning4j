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
package org.deeplearning4j.nn.modelimport.keras.layers.core;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * @author Max Pumperla
 */
public class KerasReshapeTest {

    private Integer keras1 = 1;
    private Integer keras2 = 2;
    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();
    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();


    @Test
    public void testReshapeLayer() throws Exception {
        buildReshapeLayer(conf1, keras1);
        buildReshapeLayer(conf2, keras2);
    }

    @Test
    public void testReshapeDynamicMinibatch() throws Exception {
        testDynamicMinibatches(conf1, keras1);
        testDynamicMinibatches(conf2, keras2);
    }

    private void buildReshapeLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        int[] targetShape = new int[]{10, 5};
        List<Integer> targetShapeList = new ArrayList<>();
        targetShapeList.add(targetShape[0]);
        targetShapeList.add(targetShape[1]);
        ReshapePreprocessor preProcessor = getReshapePreProcessor(conf, kerasVersion, targetShapeList);
        assertEquals(preProcessor.getTargetShape()[0], targetShape[0]);
        assertEquals(preProcessor.getTargetShape()[1], targetShape[1]);
    }

    private ReshapePreprocessor getReshapePreProcessor(KerasLayerConfiguration conf, Integer kerasVersion,
                                                       List<Integer> targetShapeList)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_RESHAPE());
        Map<String, Object> config = new HashMap<>();
        String LAYER_FIELD_TARGET_SHAPE = "target_shape";
        config.put(LAYER_FIELD_TARGET_SHAPE, targetShapeList);
        String layerName = "reshape";
        config.put(conf.getLAYER_FIELD_NAME(), layerName);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        InputType inputType = InputType.InputTypeFeedForward.feedForward(20);
        return (ReshapePreprocessor) new KerasReshape(layerConfig).getInputPreprocessor(inputType);

    }

    private void testDynamicMinibatches(KerasLayerConfiguration conf, Integer kerasVersion) throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        List<Integer> targetShape = Collections.singletonList(20);
        ReshapePreprocessor preprocessor = getReshapePreProcessor(conf, kerasVersion, targetShape);
        INDArray r1 = preprocessor.preProcess(Nd4j.zeros(10, 20), 10, LayerWorkspaceMgr.noWorkspaces());
        INDArray r2 = preprocessor.preProcess(Nd4j.zeros(5, 20), 5, LayerWorkspaceMgr.noWorkspaces());
        Assert.assertArrayEquals(r2.shape(), new int[]{5, 20});
        Assert.assertArrayEquals(r1.shape(), new int[]{10, 20});
    }
}
