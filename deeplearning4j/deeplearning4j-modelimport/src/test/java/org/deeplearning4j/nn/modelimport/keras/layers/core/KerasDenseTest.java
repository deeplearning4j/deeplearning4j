/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.modelimport.keras.layers.core;

import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasTestUtils;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import java.util.HashMap;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

/**
 * @author Max Pumperla
 */
@DisplayName("Keras Dense Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasDenseTest extends BaseDL4JTest {

    private Integer keras1 = 1;

    private Integer keras2 = 2;

    private Keras1LayerConfiguration conf1 = new Keras1LayerConfiguration();

    private Keras2LayerConfiguration conf2 = new Keras2LayerConfiguration();

    private final String ACTIVATION_KERAS = "linear";

    private final String ACTIVATION_DL4J = "identity";

    private final String LAYER_NAME = "dense";

    private final String INIT_KERAS = "glorot_normal";

    private final IWeightInit INIT_DL4J = new WeightInitXavier();

    private final double L1_REGULARIZATION = 0.01;

    private final double L2_REGULARIZATION = 0.02;

    private final double DROPOUT_KERAS = 0.3;

    private final double DROPOUT_DL4J = 1 - DROPOUT_KERAS;

    private final int N_OUT = 13;

    @Test
    @DisplayName("Test Dense Layer")
    void testDenseLayer() throws Exception {
        buildDenseLayer(conf1, keras1);
        buildDenseLayer(conf2, keras2);
    }

    private void buildDenseLayer(KerasLayerConfiguration conf, Integer kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DENSE());
        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_ACTIVATION(), ACTIVATION_KERAS);
        config.put(conf.getLAYER_FIELD_NAME(), LAYER_NAME);
        if (kerasVersion == 1) {
            config.put(conf.getLAYER_FIELD_INIT(), INIT_KERAS);
        } else {
            Map<String, Object> init = new HashMap<>();
            init.put("class_name", conf.getINIT_GLOROT_NORMAL());
            config.put(conf.getLAYER_FIELD_INIT(), init);
        }
        Map<String, Object> W_reg = new HashMap<>();
        W_reg.put(conf.getREGULARIZATION_TYPE_L1(), L1_REGULARIZATION);
        W_reg.put(conf.getREGULARIZATION_TYPE_L2(), L2_REGULARIZATION);
        config.put(conf.getLAYER_FIELD_W_REGULARIZER(), W_reg);
        config.put(conf.getLAYER_FIELD_DROPOUT(), DROPOUT_KERAS);
        config.put(conf.getLAYER_FIELD_OUTPUT_DIM(), N_OUT);
        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);
        DenseLayer layer = new KerasDense(layerConfig, false).getDenseLayer();
        assertEquals(ACTIVATION_DL4J, layer.getActivationFn().toString());
        assertEquals(LAYER_NAME, layer.getLayerName());
        assertEquals(INIT_DL4J, layer.getWeightInitFn());
        assertEquals(L1_REGULARIZATION, KerasTestUtils.getL1(layer), 0.0);
        assertEquals(L2_REGULARIZATION, KerasTestUtils.getL2(layer), 0.0);
        assertEquals(new Dropout(DROPOUT_DL4J), layer.getIDropout());
        assertEquals(N_OUT, layer.getNOut());
    }
}
