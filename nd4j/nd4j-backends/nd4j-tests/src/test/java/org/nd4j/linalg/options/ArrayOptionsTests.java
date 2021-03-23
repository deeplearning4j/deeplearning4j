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

package org.nd4j.linalg.options;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

@Slf4j
@Tag(TagNames.JAVA_ONLY)
public class ArrayOptionsTests extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayType_0(Nd4jBackend backend) {
        long[]  shapeInfo = new long[]{2, 2, 2, 2, 1, 0, 1, 99};
        assertEquals(ArrayType.DENSE, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayType_1(Nd4jBackend backend) {
        long[]  shapeInfo = new long[]{2, 2, 2, 2, 1, 0, 1, 99};
        ArrayOptionsHelper.setOptionBit(shapeInfo, ArrayType.EMPTY);

        assertEquals(ArrayType.EMPTY, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayType_2(Nd4jBackend backend) {
        long[]  shapeInfo = new long[]{2, 2, 2, 2, 1, 0, 1, 99};
        ArrayOptionsHelper.setOptionBit(shapeInfo, ArrayType.SPARSE);

        assertEquals(ArrayType.SPARSE, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArrayType_3(Nd4jBackend backend) {
        long[]  shapeInfo = new long[]{2, 2, 2, 2, 1, 0, 1, 99};
        ArrayOptionsHelper.setOptionBit(shapeInfo, ArrayType.COMPRESSED);

        assertEquals(ArrayType.COMPRESSED, ArrayOptionsHelper.arrayType(shapeInfo));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDataTypesToFromLong(Nd4jBackend backend) {

        for(DataType dt : DataType.values()){
            if(dt == DataType.UNKNOWN)
                continue;
            String s = dt.toString();
            long l = 0;
            l = ArrayOptionsHelper.setOptionBit(l, dt);
            assertNotEquals(0, l,s);
            DataType dt2 = ArrayOptionsHelper.dataType(l);
            assertEquals(dt, dt2,s);
        }

    }

    @Override
    public char ordering() {
        return 'c';
    }
}
