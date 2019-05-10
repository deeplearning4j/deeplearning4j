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

package org.deeplearning4j.nn.conf.serde;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.shade.jackson.core.JsonLocation;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.io.StringReader;
import java.util.List;

public class MultiLayerConfigurationDeserializer extends BaseNetConfigDeserializer<MultiLayerConfiguration> {

    public MultiLayerConfigurationDeserializer(JsonDeserializer<?> defaultDeserializer) {
        super(defaultDeserializer, MultiLayerConfiguration.class);
    }

    @Override
    public MultiLayerConfiguration deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        long charOffsetStart = jp.getCurrentLocation().getCharOffset();

        MultiLayerConfiguration conf = (MultiLayerConfiguration) defaultDeserializer.deserialize(jp, ctxt);
        Layer[] layers = new Layer[conf.getConfs().size()];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = conf.getConf(i).getLayer();
        }

        //Now, check if we need to manually handle IUpdater deserialization from legacy format
        boolean attemptIUpdaterFromLegacy = requiresIUpdaterFromLegacy(layers);

        boolean requiresLegacyRegularizationHandling = requiresRegularizationFromLegacy(layers);
        boolean requiresLegacyWeightInitHandling = requiresWeightInitFromLegacy(layers);

        if(attemptIUpdaterFromLegacy || requiresLegacyRegularizationHandling || requiresLegacyWeightInitHandling) {
            JsonLocation endLocation = jp.getCurrentLocation();
            long charOffsetEnd = endLocation.getCharOffset();
            Object sourceRef = endLocation.getSourceRef();
            String s;
            if (sourceRef instanceof StringReader) {
                //Workaround: sometimes sourceRef is a String, sometimes a StringReader
                ((StringReader) sourceRef).reset();
                s = IOUtils.toString((StringReader)sourceRef);
            } else {
                s = sourceRef.toString();
            }
            String jsonSubString = s.substring((int) charOffsetStart - 1, (int) charOffsetEnd);

            ObjectMapper om = NeuralNetConfiguration.mapper();
            JsonNode rootNode = om.readTree(jsonSubString);

            ArrayNode confsNode = (ArrayNode)rootNode.get("confs");

            for( int i=0; i<layers.length; i++ ){
                ObjectNode on = (ObjectNode) confsNode.get(i);
                ObjectNode confNode = null;
                if(layers[i] instanceof BaseLayer && ((BaseLayer)layers[i]).getIUpdater() == null){
                    //layer -> (first/only child) -> updater
                    if(on.has("layer")){
                        confNode = on;
                        on = (ObjectNode) on.get("layer");
                    } else {
                        continue;
                    }
                    on = (ObjectNode) on.elements().next();

                    handleUpdaterBackwardCompatibility((BaseLayer)layers[i], on);
                }

                if(attemptIUpdaterFromLegacy) {
                    if (layers[i].getIDropout() == null) {
                        //Check for legacy dropout/dropconnect
                        if (on.has("dropOut")) {
                            double d = on.get("dropOut").asDouble();
                            if (!Double.isNaN(d)) {
                                //Might be dropout or dropconnect...
                                if (confNode != null && layers[i] instanceof BaseLayer && confNode.has("useDropConnect")
                                        && confNode.get("useDropConnect").asBoolean(false)) {
                                    ((BaseLayer) layers[i]).setWeightNoise(new DropConnect(d));
                                } else {
                                    if (d > 0.0) {
                                        layers[i].setIDropout(new Dropout(d));
                                    }
                                }
                            }
                        }
                    }
                }

                if(requiresLegacyRegularizationHandling) {
                    if (layers[i] instanceof BaseLayer && ((BaseLayer) layers[i]).getRegularization() == null) {
                        if(on.has("layer")){
                            //Legacy format
                            ObjectNode layerNode = (ObjectNode)on.get("layer");
                            if(layerNode.has("@class")){
                                //Later legacy format: class field for JSON subclass
                                on = layerNode;
                            } else {
                                //Early legacy format: wrapper object for JSON subclass
                                on = (ObjectNode) on.get("layer").elements().next();
                            }
                        }
                        handleL1L2BackwardCompatibility((BaseLayer) layers[i], on);
                    }
                }

                if(requiresLegacyWeightInitHandling){
                    if (layers[i] instanceof BaseLayer && ((BaseLayer) layers[i]).getWeightInitFn() == null) {
                        if(on.has("layer")){
                            //Legacy format
                            ObjectNode layerNode = (ObjectNode)on.get("layer");
                            if(layerNode.has("@class")){
                                //Later legacy format: class field for JSON subclass
                                on = layerNode;
                            } else {
                                //Early legacy format: wrapper object for JSON subclass
                                on = (ObjectNode) on.get("layer").elements().next();
                            }
                        }
                        handleWeightInitBackwardCompatibility((BaseLayer) layers[i], on);
                    }
                }
            }
        }





        //After 1.0.0-beta3, batchnorm reparameterized to support both variance and log10stdev
        //JSON deserialization uses public BatchNormalization() constructor which defaults to log10stdev now
        // but, as there is no useLogStdev=false property for legacy batchnorm JSON, the 'real' value (useLogStdev=false)
        // is not set to override the default, unless we do it manually here
        for(NeuralNetConfiguration nnc : conf.getConfs()){
            Layer l = nnc.getLayer();
            if(l instanceof BatchNormalization){
                BatchNormalization bn = (BatchNormalization)l;
                List<String> vars = nnc.getVariables();
                boolean isVariance = vars.contains(BatchNormalizationParamInitializer.GLOBAL_VAR);
                bn.setUseLogStd(!isVariance);
            }
        }

        return conf;
    }
}
