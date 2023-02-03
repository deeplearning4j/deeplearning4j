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

package org.eclipse.deeplearning4j.nd4j.linalg.shape;


import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@NativeTag
@Tag(TagNames.NDARRAY_INDEXING)
public class LongShapeTests extends BaseNd4jTestWithBackends {



    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLongBuffer_1(Nd4jBackend backend) {
        var exp = new long[]{2, 5, 3, 3, 1, 0, 1, 99};
        var buffer = Nd4j.getDataBufferFactory().createLong(exp);

        var java = buffer.asLong();

        assertArrayEquals(exp, java);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLongShape_1(Nd4jBackend backend) {
        var exp = new long[]{2, 5, 3, 3, 1, 16384, 1, 99};

        var array = Nd4j.createUninitialized(DataType.DOUBLE, 5, 3);
        var buffer = array.shapeInfoDataBuffer();

        var java = buffer.asLong();

        assertArrayEquals(exp, java);
        assertEquals(8, buffer.getElementSize());
    }



    @Override
    public char ordering() {
        return 'c';
    }
}
