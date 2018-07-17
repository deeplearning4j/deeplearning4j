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
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;

/**
 * Simple JSON serializer for use in {@link org.nd4j.linalg.lossfunctions.ILossFunction} weight serialization.
 * Serializes an INDArray as a double[]
 *
 * @author Alex Black
 */
public class RowVectorSerializer extends JsonSerializer<INDArray> {
    @Override
    public void serialize(INDArray array, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException {
        if (array.isView()) {
            array = array.dup();
        }
        double[] dArr = array.data().asDouble();
        jsonGenerator.writeObject(dArr);
    }
}
