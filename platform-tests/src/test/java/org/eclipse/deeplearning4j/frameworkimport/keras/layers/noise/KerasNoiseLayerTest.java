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
package org.eclipse.deeplearning4j.frameworkimport.keras.layers.noise;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.dropout.AlphaDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.dropout.GaussianNoise;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.modelimport.keras.config.Keras1LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.Keras2LayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.config.KerasLayerConfiguration;
import org.deeplearning4j.nn.modelimport.keras.layers.noise.KerasAlphaDropout;
import org.deeplearning4j.nn.modelimport.keras.layers.noise.KerasGaussianDropout;
import org.deeplearning4j.nn.modelimport.keras.layers.noise.KerasGaussianNoise;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Parameterized test covering all three Keras noise/dropout layer types:
 * AlphaDropout, GaussianDropout, and GaussianNoise.
 *
 * Each variant is exercised against both Keras 1 and Keras 2 layer configurations.
 *
 * @author Max Pumperla
 */
@DisplayName("Keras Noise Layer Test")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
class KerasNoiseLayerTest extends BaseDL4JTest {

    private static final double RATE_KERAS = 0.3;
    private static final double RATE_DL4J  = 1 - RATE_KERAS;

    /**
     * Enumerates each noise layer variant with the information needed to build and
     * assert it: the Keras config field key, the numeric value to insert, and the
     * expected DL4J {@link IDropout} instance.
     */
    enum NoiseLayerVariant {

        ALPHA_DROPOUT("alpha_dropout") {
            @Override
            public void putRateField(KerasLayerConfiguration conf, Map<String, Object> config) {
                config.put(conf.getLAYER_FIELD_RATE(), RATE_KERAS);
            }

            @Override
            public DropoutLayer buildLayer(Map<String, Object> layerConfig) throws Exception {
                return new KerasAlphaDropout(layerConfig).getAlphaDropoutLayer();
            }

            @Override
            public IDropout expectedDropout() {
                return new AlphaDropout(RATE_DL4J);
            }
        },

        GAUSSIAN_DROPOUT("gaussian_dropout") {
            @Override
            public void putRateField(KerasLayerConfiguration conf, Map<String, Object> config) {
                config.put(conf.getLAYER_FIELD_RATE(), RATE_KERAS);
            }

            @Override
            public DropoutLayer buildLayer(Map<String, Object> layerConfig) throws Exception {
                return new KerasGaussianDropout(layerConfig).getGaussianDropoutLayer();
            }

            @Override
            public IDropout expectedDropout() {
                return new GaussianDropout(RATE_DL4J);
            }
        },

        GAUSSIAN_NOISE("gaussian_noise") {
            @Override
            public void putRateField(KerasLayerConfiguration conf, Map<String, Object> config) {
                // GaussianNoise uses stddev (gaussian_variance key), passed directly without inversion.
                config.put(conf.getLAYER_FIELD_GAUSSIAN_VARIANCE(), RATE_KERAS);
            }

            @Override
            public DropoutLayer buildLayer(Map<String, Object> layerConfig) throws Exception {
                return new KerasGaussianNoise(layerConfig).getGaussianNoiseLayer();
            }

            @Override
            public IDropout expectedDropout() {
                return new GaussianNoise(RATE_KERAS);
            }
        };

        private final String layerName;

        NoiseLayerVariant(String layerName) {
            this.layerName = layerName;
        }

        public String getLayerName() {
            return layerName;
        }

        /** Inserts the variant-specific parameter field into the inner config map. */
        public abstract void putRateField(KerasLayerConfiguration conf, Map<String, Object> config);

        /** Constructs the Keras wrapper and returns the parsed DL4J {@link DropoutLayer}. */
        public abstract DropoutLayer buildLayer(Map<String, Object> layerConfig) throws Exception;

        /** Returns the expected DL4J {@link IDropout} after import. */
        public abstract IDropout expectedDropout();
    }

    @ParameterizedTest(name = "{0} — keras1 & keras2")
    @EnumSource(NoiseLayerVariant.class)
    @DisplayName("Test noise layer import for both Keras versions")
    void testNoiseLayer(NoiseLayerVariant variant) throws Exception {
        buildAndAssertLayer(variant, new Keras1LayerConfiguration(), 1);
        buildAndAssertLayer(variant, new Keras2LayerConfiguration(), 2);
    }

    private void buildAndAssertLayer(NoiseLayerVariant variant,
                                     KerasLayerConfiguration conf,
                                     int kerasVersion) throws Exception {
        Map<String, Object> layerConfig = new HashMap<>();
        layerConfig.put(conf.getLAYER_FIELD_CLASS_NAME(), conf.getLAYER_CLASS_NAME_DROPOUT());

        Map<String, Object> config = new HashMap<>();
        config.put(conf.getLAYER_FIELD_NAME(), variant.getLayerName());
        variant.putRateField(conf, config);

        layerConfig.put(conf.getLAYER_FIELD_CONFIG(), config);
        layerConfig.put(conf.getLAYER_FIELD_KERAS_VERSION(), kerasVersion);

        DropoutLayer layer = variant.buildLayer(layerConfig);
        assertEquals(variant.getLayerName(), layer.getLayerName());
        assertEquals(variant.expectedDropout(), layer.getIDropout());
    }
}
