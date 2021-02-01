/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.deeplearning4j.arbiter.optimize.serde.jackson;

import org.apache.commons.codec.binary.Base64;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * A custom deserializer to be used in conjunction with {@link FixedValueSerializer}
 * @author Alex Black
 */
public class FixedValueDeserializer extends JsonDeserializer<FixedValue> {
    @Override
    public FixedValue deserialize(JsonParser p, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        JsonNode node = p.getCodec().readTree(p);
        String className = node.get("@valueclass").asText();
        Class<?> c;
        try {
            c = Class.forName(className);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        if(node.has("value")){
            //Number, String, Enum
            JsonNode valueNode = node.get("value");
            Object o = new ObjectMapper().treeToValue(valueNode, c);
            return new FixedValue<>(o);
        } else {
            //Everything else
            JsonNode valueNode = node.get("data");
            String data = valueNode.asText();

            byte[] b = new Base64().decode(data);
            ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(b));
            try {
                Object o = ois.readObject();
                return new FixedValue<>(o);
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }
    }
}
