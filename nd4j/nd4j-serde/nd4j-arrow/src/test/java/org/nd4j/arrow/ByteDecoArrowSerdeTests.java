/*******************************************************************************
 * Copyright (c) 2019 Konduit KK
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

package org.nd4j.arrow;

import org.bytedeco.arrow.ArrowBuffer;
import org.bytedeco.arrow.Tensor;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static org.junit.Assert.assertEquals;

public class ByteDecoArrowSerdeTests {


    @Test
    public void testBufferConversion() {
        for(DataType value : DataType.values()) {
            assertBufferCreation(Nd4j.createBuffer(new int[]{1,1},value,0));
        }

    }


    @Test
    public void testToTensor() {
        for(DataType value : DataType.values()) {
            INDArray arr = Nd4j.create(Nd4j.createBuffer(new int[]{1,1},value,0));
            Tensor convert = ByteDecoArrowSerde.toTensor(arr);
            INDArray convertedBack = ByteDecoArrowSerde.fromTensor(convert);
            assertEquals(arr,convertedBack);
        }
    }

    @Test
    public void testToFromTensorDataTypes() {
        for(DataType dataType : DataType.values()) {
            if(dataType == DataType.COMPRESSED || dataType == DataType.BFLOAT16 || dataType == DataType.UNKNOWN)
                continue;

            org.bytedeco.arrow.DataType dataType1 = ByteDecoArrowSerde.arrowDataTypeForNd4j(dataType);
            DataType dataType2 = ByteDecoArrowSerde.dataBufferTypeTypeForArrow(dataType1);

            assertEquals(dataType,dataType2);
        }
    }

    private void assertBufferCreation(DataBuffer buffer) {
        Pair<ArrowBuffer, org.bytedeco.arrow.DataType> arrowBuffer = ByteDecoArrowSerde.fromNd4jBuffer(buffer);
        assertEquals(buffer.dataType(),ByteDecoArrowSerde.dataBufferTypeTypeForArrow(arrowBuffer.getRight()));
        DataBuffer buffer1 = ByteDecoArrowSerde.fromArrowBuffer(arrowBuffer.getFirst(), arrowBuffer.getRight());
        assertEquals(buffer1,buffer1);
    }

}
