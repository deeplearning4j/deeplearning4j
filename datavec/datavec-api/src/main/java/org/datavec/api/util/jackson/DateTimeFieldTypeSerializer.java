/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.util.jackson;

import org.joda.time.DateTimeFieldType;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * JsonSerializer for serializing Jodatime {@link DateTimeFieldType} instances
 *
 * @author Alex Black
 */
public class DateTimeFieldTypeSerializer extends JsonSerializer<DateTimeFieldType> {
    @Override
    public void serialize(DateTimeFieldType dateTimeFieldType, JsonGenerator jsonGenerator,
                    SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
        jsonGenerator.writeStartObject();
        jsonGenerator.writeStringField("fieldType", dateTimeFieldType.getName());
        jsonGenerator.writeEndObject();
    }
}
