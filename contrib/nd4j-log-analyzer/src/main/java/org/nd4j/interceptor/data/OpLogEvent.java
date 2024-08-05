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

import lombok.*;
import org.json.JSONArray;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.IOException;
import java.util.*;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
@Setter
@ToString
public class OpLogEvent {
    public String opName;

    @Builder.Default
    @JsonSerialize(using = InputOutputSerializer.class)
    public Map<Integer,String> inputs = new LinkedHashMap<>();

    @Builder.Default
    @JsonSerialize(using = InputOutputSerializer.class)
    public Map<Integer,String> outputs = new LinkedHashMap();

    @JsonSerialize(using = StackTraceSerializer.class)
    public String stackTrace;

    public String firstNonExecutionCodeLine;



    public long eventId;


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OpLogEvent that = (OpLogEvent) o;
        return Objects.equals(firstNonExecutionCodeLine, that.firstNonExecutionCodeLine) &&
                Objects.equals(opName, that.opName) &&
                Objects.equals(inputs, that.inputs) &&
                Objects.equals(outputs, that.outputs);
    }

    @Override
    public int hashCode() {
        return Objects.hash(firstNonExecutionCodeLine, opName, inputs, outputs);
    }

    public static class InputOutputSerializer extends JsonSerializer<Map<Integer, String>> {
        @Override
        public void serialize(Map<Integer, String> value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.registerModule(new JSONArraySerializer.JSONArraySerializerModule());
            objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
            Map<Integer, Object> write = new LinkedHashMap<>();
            for (Map.Entry<Integer, String> entry : value.entrySet()) {
                Integer key = entry.getKey();
                String item = entry.getValue();
                try {
                    JSONArray jsonArray = new JSONArray(item);
                    List<Object> innerList = new ArrayList<>();
                    for (int i = 0; i < jsonArray.length(); i++) {
                        Object innerItem = jsonArray.get(i);
                        if (innerItem instanceof JSONArray) {
                            JSONArray innerArray = (JSONArray) innerItem;
                            List<Object> innerArrayList = new ArrayList<>();
                            for (int j = 0; j < innerArray.length(); j++) {
                                innerArrayList.add(innerArray.get(j));
                            }
                            innerList.add(innerArrayList);
                        } else {
                            innerList.add(innerItem);
                        }
                    }
                    write.put(key, innerList);
                } catch (Exception e) {
                    // scalar cases
                    write.put(key, item);
                }
            }
            gen.writeStartObject();
            for (Map.Entry<Integer, Object> entry : write.entrySet()) {
                gen.writeFieldName(entry.getKey().toString());
                Object item = entry.getValue();
                if (item instanceof List) {
                    gen.writeStartArray();
                    for (Object innerItem : (List<?>) item) {
                        if (innerItem instanceof List) {
                            gen.writeStartArray();
                            for (Object innerArrayItem : (List<?>) innerItem) {
                                gen.writeObject(innerArrayItem);
                            }
                            gen.writeEndArray();
                        } else {
                            gen.writeObject(innerItem);
                        }
                    }
                    gen.writeEndArray();
                } else {
                    gen.writeString((String) item);
                }
            }
            gen.writeEndObject();
        }
    }
    public static class StackTraceSerializer extends JsonSerializer<String> {
        @Override
        public void serialize(String value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
            JSONArray jsonArray = new JSONArray();
            for(String item : value.split("\n")) {
                jsonArray.put(item.replace("\"\"",""));
            }
            gen.writeRawValue(jsonArray.toString(2));
        }
    }
}