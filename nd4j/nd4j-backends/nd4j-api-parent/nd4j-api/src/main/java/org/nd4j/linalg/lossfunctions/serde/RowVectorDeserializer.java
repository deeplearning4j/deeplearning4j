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

package org.nd4j.linalg.lossfunctions.serde;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;

/**
 * Simple JSON deserializer for use in {@link org.nd4j.linalg.lossfunctions.ILossFunction} loss function weight serialization.
 * Used in conjunction with {@link RowVectorSerializer}
 *
 * @author Alex Black
 */
public class RowVectorDeserializer extends JsonDeserializer<INDArray> {
    @Override
    public INDArray deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
                    throws IOException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        if (node == null)
            return null;
        int size = node.size();
        double[] d = new double[size];
        for (int i = 0; i < size; i++) {
            d[i] = node.get(i).asDouble();
        }

        return Nd4j.create(d);
    }
}
