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

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.nd4j.shade.jackson.core.JsonLocation;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;

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


        if(attemptIUpdaterFromLegacy) {
            JsonLocation endLocation = jp.getCurrentLocation();
            long charOffsetEnd = endLocation.getCharOffset();
            String jsonSubString = endLocation.getSourceRef().toString().substring((int) charOffsetStart - 1, (int) charOffsetEnd);

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

                if(layers[i].getIDropout() == null){
                    //Check for legacy dropout/dropconnect
                    if(on.has("dropOut")){
                        double d = on.get("dropOut").asDouble();
                        if(!Double.isNaN(d)){
                            //Might be dropout or dropconnect...
                            if(confNode != null && layers[i] instanceof BaseLayer && confNode.has("useDropConnect")
                                    && confNode.get("useDropConnect").asBoolean(false)){
                                ((BaseLayer)layers[i]).setWeightNoise(new DropConnect(d));
                            } else {
                                if(d > 0.0) {
                                    layers[i].setIDropout(new Dropout(d));
                                }
                            }
                        }
                    }
                }
            }
        }


        return conf;
    }
}
