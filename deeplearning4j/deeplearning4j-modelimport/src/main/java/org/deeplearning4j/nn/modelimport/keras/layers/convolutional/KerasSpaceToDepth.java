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

package org.deeplearning4j.nn.modelimport.keras.layers.convolutional;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.SpaceToDepthLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasLayerUtils;

import java.util.Map;

public class KerasSpaceToDepth extends KerasLayer {

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig dictionary containing Keras layer configuration.
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration exception
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration exception
     */
    public KerasSpaceToDepth(Map<String, Object> layerConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        this(layerConfig, true);
    }

    /**
     * Constructor from parsed Keras layer configuration dictionary.
     *
     * @param layerConfig           dictionary containing Keras layer configuration
     * @param enforceTrainingConfig whether to enforce training-related configuration options
     * @throws InvalidKerasConfigurationException     Invalid Keras configuration exception
     * @throws UnsupportedKerasConfigurationException Unsupported Keras configuration exception
     */
    public KerasSpaceToDepth(Map<String, Object> layerConfig, boolean enforceTrainingConfig)
            throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        super(layerConfig, enforceTrainingConfig);

        // Read block_size from inner config if available (native SpaceToDepth layers store it there).
        // For Lambda-wrapped layers, block_size is inside the serialized lambda and cannot be read,
        // so we fall back to 2 (used by YOLO9000).
        Map<String, Object> innerConfig = KerasLayerUtils.getInnerLayerConfigFromConfig(layerConfig, conf);
        int blockSize = 2;
        if (innerConfig.containsKey("block_size")) {
            blockSize = ((Number) innerConfig.get("block_size")).intValue();
        }

        SpaceToDepthLayer.Builder builder = new SpaceToDepthLayer.Builder()
                .blocks(blockSize)
                //the default data format is tensorflow/NWHC for keras import
                .dataFormat(SpaceToDepthLayer.DataFormat.NHWC)
                .name(layerName);

        this.layer = builder.build();
        this.vertex = null;
    }

    /**
     * Get DL4J SpaceToDepth layer.
     *
     * @return SpaceToDepth layer
     */
    public SpaceToDepthLayer getSpaceToDepthLayer() {
        return (SpaceToDepthLayer) this.layer;
    }

    /**
     * Get layer output type.
     *
     * @param inputType Array of InputTypes
     * @return output type as InputType
     * @throws InvalidKerasConfigurationException Invalid Keras config
     */
    @Override
    public InputType getOutputType(InputType... inputType) throws InvalidKerasConfigurationException {
        if (inputType.length > 1)
            throw new InvalidKerasConfigurationException(
                    "Keras Space to Depth layer accepts only one input (received " + inputType.length + ")");
        return this.getSpaceToDepthLayer().getOutputType(-1, inputType[0]);
    }

}