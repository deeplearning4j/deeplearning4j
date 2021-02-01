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

import org.apache.commons.net.util.Base64;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.type.WritableTypeId;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.shade.jackson.databind.jsontype.TypeSerializer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

import static org.nd4j.shade.jackson.core.JsonToken.START_OBJECT;

/**
 * A custom serializer to handle arbitrary object types
 * Uses standard JSON where safe (number, string, enumerations) or Java object serialization (bytes -> base64)
 * The latter is not an ideal approach, but Jackson doesn't support serialization/deserialization of arbitrary
 * objects very well
 *
 * @author Alex Black
 */
public class FixedValueSerializer extends JsonSerializer<FixedValue> {
    @Override
    public void serialize(FixedValue fixedValue, JsonGenerator j, SerializerProvider serializerProvider) throws IOException {
        Object o = fixedValue.getValue();

        j.writeStringField("@valueclass", o.getClass().getName());
        if(o instanceof Number || o instanceof String || o instanceof Enum){
            j.writeObjectField("value", o);
        } else {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(o);
            baos.close();
            byte[] b = baos.toByteArray();
            String base64 = new Base64().encodeToString(b);
            j.writeStringField("data", base64);
        }
    }

    @Override
    public void serializeWithType(FixedValue value, JsonGenerator gen, SerializerProvider serializers, TypeSerializer typeSer) throws IOException {
        WritableTypeId typeId = typeSer.typeId(value, START_OBJECT);
        typeSer.writeTypePrefix(gen, typeId);
        serialize(value, gen, serializers);
        typeSer.writeTypeSuffix(gen, typeId);
    }
}
