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
package org.nd4j.interceptor.data;

import org.json.JSONArray;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.shade.jackson.databind.module.SimpleModule;

import java.io.IOException;

public class JSONArraySerializer extends JsonSerializer<JSONArray> {

    @Override
    public void serialize(JSONArray value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
        gen.writeStartArray();
        for (int i = 0; i < value.length(); i++) {
            Object item = value.opt(i);
            if (item == null) {
                gen.writeNull();
            } else if (item instanceof Boolean) {
                gen.writeBoolean((Boolean) item);
            } else if (item instanceof Integer) {
                gen.writeNumber((Integer) item);
            } else if (item instanceof Long) {
                gen.writeNumber((Long) item);
            } else if (item instanceof Double) {
                gen.writeNumber((Double) item);
            } else if (item instanceof String) {
                gen.writeString((String) item);
            } else if (item instanceof JSONArray) {
                serialize((JSONArray) item, gen, serializers);
            } else {
                gen.writeObject(item.toString());
            }
        }
        gen.writeEndArray();
    }

    public static class JSONArraySerializerModule extends SimpleModule {
        public JSONArraySerializerModule() {
            addSerializer(JSONArray.class, new JSONArraySerializer());
        }
    }
}