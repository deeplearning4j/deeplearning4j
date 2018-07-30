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

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.nd4j.shade.jackson.core.JsonLocation;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;


public class ComputationGraphConfigurationDeserializer
                extends BaseNetConfigDeserializer<ComputationGraphConfiguration> {

    public ComputationGraphConfigurationDeserializer(JsonDeserializer<?> defaultDeserializer) {
        super(defaultDeserializer, ComputationGraphConfiguration.class);
    }

    @Override
    public ComputationGraphConfiguration deserialize(JsonParser jp, DeserializationContext ctxt) throws IOException {
        long charOffsetStart = jp.getCurrentLocation().getCharOffset();
        ComputationGraphConfiguration conf = (ComputationGraphConfiguration) defaultDeserializer.deserialize(jp, ctxt);


        //Updater configuration changed after 0.8.0 release
        //Previously: enumerations and fields. Now: classes
        //Here, we manually create the appropriate Updater instances, if the IUpdater field is empty

        List<Layer> layerList = new ArrayList<>();
        Map<String, GraphVertex> vertices = conf.getVertices();
        for (Map.Entry<String, GraphVertex> entry : vertices.entrySet()) {
            if (entry.getValue() instanceof LayerVertex) {
                LayerVertex lv = (LayerVertex) entry.getValue();
                layerList.add(lv.getLayerConf().getLayer());
            }
        }

        Layer[] layers = layerList.toArray(new Layer[layerList.size()]);
        //Now, check if we need to manually handle IUpdater deserialization from legacy format
        boolean attemptIUpdaterFromLegacy = requiresIUpdaterFromLegacy(layers);

        if(attemptIUpdaterFromLegacy) {
            JsonLocation endLocation = jp.getCurrentLocation();
            long charOffsetEnd = endLocation.getCharOffset();
            String jsonSubString = endLocation.getSourceRef().toString().substring((int) charOffsetStart - 1, (int) charOffsetEnd);

            ObjectMapper om = NeuralNetConfiguration.mapper();
            JsonNode rootNode = om.readTree(jsonSubString);

            ObjectNode verticesNode = (ObjectNode) rootNode.get("vertices");
            Iterator<JsonNode> iter = verticesNode.elements();
            int layerIdx = 0;
            while(iter.hasNext()){
                JsonNode next = iter.next();
                ObjectNode confNode = null;
                if(next.has("LayerVertex")){
                    next = next.get("LayerVertex");
                    if(next.has("layerConf")){
                        confNode = (ObjectNode) next.get("layerConf");
                        next = confNode.get("layer").elements().next();
                    } else {
                        continue;
                    }

                    if(layers[layerIdx] instanceof BaseLayer && ((BaseLayer)layers[layerIdx]).getIUpdater() == null){
                        handleUpdaterBackwardCompatibility((BaseLayer)layers[layerIdx], (ObjectNode)next);
                    }

                    if(layers[layerIdx].getIDropout() == null){
                        //Check for legacy dropout
                        if(next.has("dropOut")){
                            double d = next.get("dropOut").asDouble();
                            if(!Double.isNaN(d)){
                                //Might be dropout or dropconnect...
                                if(layers[layerIdx] instanceof BaseLayer && confNode.has("useDropConnect")
                                        && confNode.get("useDropConnect").asBoolean(false)){
                                    ((BaseLayer)layers[layerIdx]).setWeightNoise(new DropConnect(d));
                                } else {
                                    layers[layerIdx].setIDropout(new Dropout(d));
                                }
                            }
                        }
                    }

                    layerIdx++;
                }
            }
        }

        return conf;
    }
}
