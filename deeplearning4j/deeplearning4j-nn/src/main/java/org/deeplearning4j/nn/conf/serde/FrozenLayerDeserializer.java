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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.serde.legacyformat.LegacyLayerDeserializer;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * A custom deserializer for handling Frozen layers
 * This is used to handle the 2 different Layer JSON formats - old/legacy, and current
 *
 * @author Alex Black
 */
public class FrozenLayerDeserializer extends JsonDeserializer<Layer> {
    @Override
    public Layer deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException {
        JsonNode n = jp.getCodec().readTree(jp);
        JsonNode layer = n.get("layer");
        boolean newFormat = layer.has("@class");

        String internalText = layer.toString();
        Layer internal;
        if(newFormat){
            //Standard/new format
            internal = NeuralNetConfiguration.mapper().readValue(internalText, Layer.class);
        } else {
            //Legacy format
            JsonFactory factory = new JsonFactory();
            JsonParser  parser  = factory.createParser(internalText);
            parser.setCodec(jp.getCodec());
            internal = new LegacyLayerDeserializer().deserialize(parser, deserializationContext);
        }
        return new FrozenLayer(internal);
    }
}
