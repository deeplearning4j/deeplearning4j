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

package org.eclipse.deeplearning4j.ggml.format;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GGMLDataType Test")
class GGMLDataTypeTest {

    @Test
    @DisplayName("Test fromTypeId for standard types")
    void testFromTypeIdStandardTypes() {
        assertEquals(GGMLDataType.GGML_TYPE_F32, GGMLDataType.fromTypeId(0));
        assertEquals(GGMLDataType.GGML_TYPE_F16, GGMLDataType.fromTypeId(1));
        assertEquals(GGMLDataType.GGML_TYPE_Q4_0, GGMLDataType.fromTypeId(2));
        assertEquals(GGMLDataType.GGML_TYPE_Q4_1, GGMLDataType.fromTypeId(3));
        assertEquals(GGMLDataType.GGML_TYPE_Q8_0, GGMLDataType.fromTypeId(8));
    }

    @Test
    @DisplayName("Test fromTypeId for K-quant types")
    void testFromTypeIdKQuantTypes() {
        assertEquals(GGMLDataType.GGML_TYPE_Q2_K, GGMLDataType.fromTypeId(10));
        assertEquals(GGMLDataType.GGML_TYPE_Q3_K, GGMLDataType.fromTypeId(11));
        assertEquals(GGMLDataType.GGML_TYPE_Q4_K, GGMLDataType.fromTypeId(12));
        assertEquals(GGMLDataType.GGML_TYPE_Q5_K, GGMLDataType.fromTypeId(13));
        assertEquals(GGMLDataType.GGML_TYPE_Q6_K, GGMLDataType.fromTypeId(14));
    }

    @Test
    @DisplayName("Test fromTypeId for integer types")
    void testFromTypeIdIntegerTypes() {
        assertEquals(GGMLDataType.GGML_TYPE_I8, GGMLDataType.fromTypeId(25));
        assertEquals(GGMLDataType.GGML_TYPE_I16, GGMLDataType.fromTypeId(26));
        assertEquals(GGMLDataType.GGML_TYPE_I32, GGMLDataType.fromTypeId(27));
        assertEquals(GGMLDataType.GGML_TYPE_I64, GGMLDataType.fromTypeId(28));
    }

    @Test
    @DisplayName("Test fromTypeId throws for invalid type")
    void testFromTypeIdInvalid() {
        assertThrows(IllegalArgumentException.class, () -> GGMLDataType.fromTypeId(999));
    }

    @Test
    @DisplayName("Test isValidTypeId")
    void testIsValidTypeId() {
        assertTrue(GGMLDataType.isValidTypeId(0));  // F32
        assertTrue(GGMLDataType.isValidTypeId(1));  // F16
        assertTrue(GGMLDataType.isValidTypeId(8));  // Q8_0
        assertTrue(GGMLDataType.isValidTypeId(12)); // Q4_K
        assertFalse(GGMLDataType.isValidTypeId(999));
        assertFalse(GGMLDataType.isValidTypeId(-1));
    }

    @Test
    @DisplayName("Test ND4J type mapping for non-quantized types")
    void testNd4jTypeMapping() {
        assertEquals(DataType.FLOAT, GGMLDataType.GGML_TYPE_F32.getNd4jType());
        assertEquals(DataType.HALF, GGMLDataType.GGML_TYPE_F16.getNd4jType());
        assertEquals(DataType.DOUBLE, GGMLDataType.GGML_TYPE_F64.getNd4jType());
        assertEquals(DataType.BFLOAT16, GGMLDataType.GGML_TYPE_BF16.getNd4jType());
        assertEquals(DataType.BYTE, GGMLDataType.GGML_TYPE_I8.getNd4jType());
        assertEquals(DataType.SHORT, GGMLDataType.GGML_TYPE_I16.getNd4jType());
        assertEquals(DataType.INT, GGMLDataType.GGML_TYPE_I32.getNd4jType());
        assertEquals(DataType.LONG, GGMLDataType.GGML_TYPE_I64.getNd4jType());
    }

    @Test
    @DisplayName("Test quantized types return null for ND4J type")
    void testQuantizedTypesReturnNull() {
        assertNull(GGMLDataType.GGML_TYPE_Q4_0.getNd4jType());
        assertNull(GGMLDataType.GGML_TYPE_Q4_1.getNd4jType());
        assertNull(GGMLDataType.GGML_TYPE_Q8_0.getNd4jType());
        assertNull(GGMLDataType.GGML_TYPE_Q4_K.getNd4jType());
    }

    @Test
    @DisplayName("Test isQuantized")
    void testIsQuantized() {
        assertFalse(GGMLDataType.GGML_TYPE_F32.isQuantized());
        assertFalse(GGMLDataType.GGML_TYPE_F16.isQuantized());
        assertFalse(GGMLDataType.GGML_TYPE_I32.isQuantized());
        assertTrue(GGMLDataType.GGML_TYPE_Q4_0.isQuantized());
        assertTrue(GGMLDataType.GGML_TYPE_Q8_0.isQuantized());
        assertTrue(GGMLDataType.GGML_TYPE_Q4_K.isQuantized());
    }

    @Test
    @DisplayName("Test block size")
    void testBlockSize() {
        assertEquals(1, GGMLDataType.GGML_TYPE_F32.getBlockSize());
        assertEquals(1, GGMLDataType.GGML_TYPE_F16.getBlockSize());
        assertEquals(32, GGMLDataType.GGML_TYPE_Q4_0.getBlockSize());
        assertEquals(32, GGMLDataType.GGML_TYPE_Q8_0.getBlockSize());
        assertEquals(256, GGMLDataType.GGML_TYPE_Q4_K.getBlockSize());
    }

    @Test
    @DisplayName("Test bytes per element")
    void testBytesPerElement() {
        assertEquals(4.0, GGMLDataType.GGML_TYPE_F32.getBytesPerElement(), 0.001);
        assertEquals(2.0, GGMLDataType.GGML_TYPE_F16.getBytesPerElement(), 0.001);
        assertEquals(8.0, GGMLDataType.GGML_TYPE_F64.getBytesPerElement(), 0.001);
        assertEquals(1.0, GGMLDataType.GGML_TYPE_I8.getBytesPerElement(), 0.001);
    }

    @Test
    @DisplayName("Test storage bytes calculation for non-quantized")
    void testCalculateStorageBytesNonQuantized() {
        assertEquals(400, GGMLDataType.GGML_TYPE_F32.calculateStorageBytes(100));
        assertEquals(200, GGMLDataType.GGML_TYPE_F16.calculateStorageBytes(100));
        assertEquals(100, GGMLDataType.GGML_TYPE_I8.calculateStorageBytes(100));
    }

    @Test
    @DisplayName("Test all type IDs are unique")
    void testUniqueTypeIds() {
        GGMLDataType[] types = GGMLDataType.values();
        for (int i = 0; i < types.length; i++) {
            for (int j = i + 1; j < types.length; j++) {
                assertNotEquals(types[i].getTypeId(), types[j].getTypeId(),
                        "Type IDs should be unique: " + types[i] + " and " + types[j]);
            }
        }
    }
}
