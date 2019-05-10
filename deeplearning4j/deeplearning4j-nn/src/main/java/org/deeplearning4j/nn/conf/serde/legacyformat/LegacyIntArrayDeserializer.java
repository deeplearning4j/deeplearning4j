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

package org.deeplearning4j.nn.conf.serde.legacyformat;

import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import java.io.IOException;

/**
 * Deserialize either an int[] to an int[], or a single int x to int[]{x,x}
 *
 * Used when supporting a configuration format change from single int value to int[], as for Upsampling2D
 * between 1.0.0-alpha and 1.0.0-beta
 *
 * @author Alex Black
 */
public class LegacyIntArrayDeserializer extends JsonDeserializer<int[]> {
    @Override
    public int[] deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        JsonNode n = jp.getCodec().readTree(jp);
        if(n.isArray()){
            ArrayNode an = (ArrayNode)n;
            int size = an.size();
            int[] out = new int[size];
            for( int i=0; i<size; i++ ){
                out[i] = an.get(i).asInt();
            }
            return out;
        } else if(n.isNumber()){
            int v = n.asInt();
            return new int[]{v,v};
        } else {
            throw new IllegalStateException("Could not deserialize value: " + n);
        }
    }
}
