/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.serde;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
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
import org.nd4j.shade.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.io.StringReader;
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
        boolean requireLegacyRegularizationHandling = requiresRegularizationFromLegacy(layers);
        boolean requiresLegacyWeightInitHandling = requiresWeightInitFromLegacy(layers);
        boolean requiresLegacyActivationHandling = requiresActivationFromLegacy(layers);
        boolean requiresLegacyLossHandling = requiresLegacyLossHandling(layers);

        Long charOffsetEnd = null;
        JsonLocation endLocation = null;
        String jsonSubString = null;
        if(attemptIUpdaterFromLegacy || requireLegacyRegularizationHandling || requiresLegacyWeightInitHandling) {
            endLocation = jp.getCurrentLocation();
            charOffsetEnd = endLocation.getCharOffset();
            Object sourceRef = endLocation.getSourceRef();
            String s;
            if (sourceRef instanceof StringReader) {
                //Workaround: sometimes sourceRef is a String, sometimes a StringReader
                ((StringReader) sourceRef).reset();
                s = IOUtils.toString((StringReader)sourceRef);
            } else {
                s = sourceRef.toString();
            }
            jsonSubString = s.substring((int) charOffsetStart - 1, charOffsetEnd.intValue());

            ObjectMapper om = NeuralNetConfiguration.mapper();
            JsonNode rootNode = om.readTree(jsonSubString);

            ObjectNode verticesNode = (ObjectNode) rootNode.get("vertices");
            Iterator<JsonNode> iter = verticesNode.elements();
            int layerIdx = 0;
            while(iter.hasNext()){
                JsonNode next = iter.next();
                ObjectNode confNode = null;
                String cls = next.has("@class") ? next.get("@class").asText() : null;
                if(next.has("LayerVertex")){
                    next = next.get("LayerVertex");
                    if(next.has("layerConf")){
                        confNode = (ObjectNode) next.get("layerConf");
                        next = confNode.get("layer").elements().next();
                    } else {
                        continue;
                    }

                    if(attemptIUpdaterFromLegacy && layers[layerIdx] instanceof BaseLayer && ((BaseLayer)layers[layerIdx]).getIUpdater() == null){
                        handleUpdaterBackwardCompatibility((BaseLayer)layers[layerIdx], (ObjectNode)next);
                    }

                    if(requireLegacyRegularizationHandling && layers[layerIdx] instanceof BaseLayer && ((BaseLayer)layers[layerIdx]).getRegularization() == null){
                        handleL1L2BackwardCompatibility((BaseLayer)layers[layerIdx], (ObjectNode)next);
                    }

                    if(requiresLegacyWeightInitHandling && layers[layerIdx] instanceof BaseLayer && ((BaseLayer)layers[layerIdx]).getWeightInitFn() == null){
                        handleWeightInitBackwardCompatibility((BaseLayer)layers[layerIdx], (ObjectNode)next);
                    }

                    if(requiresLegacyActivationHandling && layers[layerIdx] instanceof BaseLayer && ((BaseLayer)layers[layerIdx]).getActivationFn() == null){
                        handleActivationBackwardCompatibility((BaseLayer)layers[layerIdx], (ObjectNode)next);
                    }

                    if(requiresLegacyLossHandling && layers[layerIdx] instanceof BaseOutputLayer && ((BaseOutputLayer)layers[layerIdx]).getLossFn() == null){
                        handleLossBackwardCompatibility((BaseOutputLayer) layers[layerIdx],  (ObjectNode)next);
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
                } else if("org.deeplearning4j.nn.conf.graph.LayerVertex".equals(cls)){
                    if(requiresLegacyWeightInitHandling && layers[layerIdx] instanceof BaseLayer && ((BaseLayer)layers[layerIdx]).getWeightInitFn() == null) {
                        //Post JSON format change for subclasses, but before WeightInit was made a class
                        confNode = (ObjectNode) next.get("layerConf");
                        next = confNode.get("layer");
                        handleWeightInitBackwardCompatibility((BaseLayer) layers[layerIdx], (ObjectNode) next);
                    }
                    layerIdx++;
                }
            }
        }

        //After 1.0.0-beta3, batchnorm reparameterized to support both variance and log10stdev
        //JSON deserialization uses public BatchNormalization() constructor which defaults to log10stdev now
        // but, as there is no useLogStdev=false property for legacy batchnorm JSON, the 'real' value (useLogStdev=false)
        // is not set to override the default, unless we do it manually here
        for(GraphVertex gv : conf.getVertices().values()){
            if(gv instanceof LayerVertex && ((LayerVertex) gv).getLayerConf().getLayer() instanceof BatchNormalization){
                BatchNormalization bn = (BatchNormalization) ((LayerVertex) gv).getLayerConf().getLayer();
                List<String> vars = ((LayerVertex) gv).getLayerConf().getVariables();
                boolean isVariance = vars.contains(BatchNormalizationParamInitializer.GLOBAL_VAR);
                bn.setUseLogStd(!isVariance);
            }
        }

        return conf;
    }
}
