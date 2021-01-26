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
package org.deeplearning4j.nn.conf.layers.objdetect;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Custom deserializer to handle change in format between beta6 (and earlier) and later versions
 *
 * @author Alex Black
 */
public class BoundingBoxesDeserializer extends JsonDeserializer<INDArray> {
    @Override
    public INDArray deserialize(JsonParser jp, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        JsonNode node = jp.getCodec().readTree(jp);
        if(node.has("dataBuffer")){
            //Must be legacy format serialization
            JsonNode arr = node.get("dataBuffer");
            int rank = node.get("rankField").asInt();
            int numElements = node.get("numElements").asInt();
            int offset = node.get("offsetField").asInt();
            JsonNode shape = node.get("shapeField");
            JsonNode stride = node.get("strideField");
            int[] shapeArr = new int[rank];
            int[] strideArr = new int[rank];
            DataBuffer buff = Nd4j.createBuffer(numElements);
            for (int i = 0; i < numElements; i++) {
                buff.put(i, arr.get(i).asDouble());
            }

            String ordering = node.get("orderingField").asText();
            for (int i = 0; i < rank; i++) {
                shapeArr[i] = shape.get(i).asInt();
                strideArr[i] = stride.get(i).asInt();
            }

            return Nd4j.create(buff, shapeArr, strideArr, offset, ordering.charAt(0));
        }
        //Standard/new format
        return new NDArrayTextDeSerializer().deserialize(node);
    }
}
