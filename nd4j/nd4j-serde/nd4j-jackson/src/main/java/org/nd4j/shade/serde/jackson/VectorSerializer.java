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

package org.nd4j.shade.serde.jackson;


import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * @author Adam Gibson
 */
public class VectorSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray indArray, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException {
        if (indArray.isView())
            indArray = indArray.dup(indArray.ordering());
        jsonGenerator.writeStartObject();
        DataBuffer view = indArray.data();
        jsonGenerator.writeArrayFieldStart("dataBuffer");
        for (int i = 0; i < view.length(); i++) {
            jsonGenerator.writeNumber(view.getDouble(i));
        }

        jsonGenerator.writeEndArray();

        jsonGenerator.writeArrayFieldStart("shapeField");
        for (int i = 0; i < indArray.rank(); i++) {
            jsonGenerator.writeNumber(indArray.size(i));
        }
        jsonGenerator.writeEndArray();

        jsonGenerator.writeArrayFieldStart("strideField");
        for (int i = 0; i < indArray.rank(); i++)
            jsonGenerator.writeNumber(indArray.stride(i));
        jsonGenerator.writeEndArray();

        jsonGenerator.writeNumberField("offsetField", indArray.offset());
        jsonGenerator.writeNumberField("rankField", indArray.rank());
        jsonGenerator.writeNumberField("numElements", view.length());
        jsonGenerator.writeStringField("orderingField", String.valueOf(indArray.ordering()));
        jsonGenerator.writeEndObject();
    }
}
