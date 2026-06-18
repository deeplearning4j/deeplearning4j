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

// TODO: Move to platform-tests/ per project convention — all tests must run from platform-tests

package org.nd4j.torchscript.format;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TorchScriptDataType Test")
class TorchScriptDataTypeTest {

    @Test
    @DisplayName("Test fromTorchName for standard types")
    void testFromTorchNameStandardTypes() {
        assertEquals(TorchScriptDataType.FLOAT32, TorchScriptDataType.fromTorchName("torch.float32"));
        assertEquals(TorchScriptDataType.FLOAT16, TorchScriptDataType.fromTorchName("torch.float16"));
        assertEquals(TorchScriptDataType.FLOAT64, TorchScriptDataType.fromTorchName("torch.float64"));
        assertEquals(TorchScriptDataType.INT32, TorchScriptDataType.fromTorchName("torch.int32"));
        assertEquals(TorchScriptDataType.INT64, TorchScriptDataType.fromTorchName("torch.int64"));
        assertEquals(TorchScriptDataType.BOOL, TorchScriptDataType.fromTorchName("torch.bool"));
    }

    @Test
    @DisplayName("Test fromTorchName for short names")
    void testFromTorchNameShortNames() {
        assertEquals(TorchScriptDataType.FLOAT32, TorchScriptDataType.fromTorchName("float32"));
        assertEquals(TorchScriptDataType.FLOAT16, TorchScriptDataType.fromTorchName("float16"));
        assertEquals(TorchScriptDataType.INT64, TorchScriptDataType.fromTorchName("int64"));
    }

    @Test
    @DisplayName("Test fromTorchName for aliases")
    void testFromTorchNameAliases() {
        assertEquals(TorchScriptDataType.FLOAT32, TorchScriptDataType.fromTorchName("torch.float"));
        assertEquals(TorchScriptDataType.FLOAT64, TorchScriptDataType.fromTorchName("torch.double"));
        assertEquals(TorchScriptDataType.FLOAT16, TorchScriptDataType.fromTorchName("torch.half"));
        assertEquals(TorchScriptDataType.INT32, TorchScriptDataType.fromTorchName("torch.int"));
        assertEquals(TorchScriptDataType.INT64, TorchScriptDataType.fromTorchName("torch.long"));
        assertEquals(TorchScriptDataType.INT16, TorchScriptDataType.fromTorchName("torch.short"));
    }

    @Test
    @DisplayName("Test fromTorchName returns null for unknown type")
    void testFromTorchNameUnknown() {
        assertNull(TorchScriptDataType.fromTorchName("unknown_type"));
        assertNull(TorchScriptDataType.fromTorchName("torch.unknown"));
    }

    @Test
    @DisplayName("Test fromSafetensorsType")
    void testFromSafetensorsType() {
        assertEquals(TorchScriptDataType.FLOAT32, TorchScriptDataType.fromSafetensorsType("F32"));
        assertEquals(TorchScriptDataType.FLOAT16, TorchScriptDataType.fromSafetensorsType("F16"));
        assertEquals(TorchScriptDataType.BFLOAT16, TorchScriptDataType.fromSafetensorsType("BF16"));
        assertEquals(TorchScriptDataType.FLOAT64, TorchScriptDataType.fromSafetensorsType("F64"));
        assertEquals(TorchScriptDataType.INT8, TorchScriptDataType.fromSafetensorsType("I8"));
        assertEquals(TorchScriptDataType.INT16, TorchScriptDataType.fromSafetensorsType("I16"));
        assertEquals(TorchScriptDataType.INT32, TorchScriptDataType.fromSafetensorsType("I32"));
        assertEquals(TorchScriptDataType.INT64, TorchScriptDataType.fromSafetensorsType("I64"));
        assertEquals(TorchScriptDataType.UINT8, TorchScriptDataType.fromSafetensorsType("U8"));
        assertEquals(TorchScriptDataType.BOOL, TorchScriptDataType.fromSafetensorsType("BOOL"));
    }

    @Test
    @DisplayName("Test fromSafetensorsType returns null for unknown")
    void testFromSafetensorsTypeUnknown() {
        assertNull(TorchScriptDataType.fromSafetensorsType("UNKNOWN"));
        assertNull(TorchScriptDataType.fromSafetensorsType("F128"));
    }

    @Test
    @DisplayName("Test ND4J type mapping")
    void testNd4jTypeMapping() {
        assertEquals(DataType.FLOAT, TorchScriptDataType.FLOAT32.getNd4jType());
        assertEquals(DataType.HALF, TorchScriptDataType.FLOAT16.getNd4jType());
        assertEquals(DataType.BFLOAT16, TorchScriptDataType.BFLOAT16.getNd4jType());
        assertEquals(DataType.DOUBLE, TorchScriptDataType.FLOAT64.getNd4jType());
        assertEquals(DataType.INT8, TorchScriptDataType.INT8.getNd4jType());
        assertEquals(DataType.INT16, TorchScriptDataType.INT16.getNd4jType());
        assertEquals(DataType.INT32, TorchScriptDataType.INT32.getNd4jType());
        assertEquals(DataType.INT64, TorchScriptDataType.INT64.getNd4jType());
        assertEquals(DataType.UINT8, TorchScriptDataType.UINT8.getNd4jType());
        assertEquals(DataType.BOOL, TorchScriptDataType.BOOL.getNd4jType());
    }

    @Test
    @DisplayName("Test complex types not supported")
    void testComplexTypesNotSupported() {
        assertNull(TorchScriptDataType.COMPLEX64.getNd4jType());
        assertNull(TorchScriptDataType.COMPLEX128.getNd4jType());
        assertFalse(TorchScriptDataType.COMPLEX64.isSupported());
        assertFalse(TorchScriptDataType.COMPLEX128.isSupported());
    }

    @Test
    @DisplayName("Test isSupported")
    void testIsSupported() {
        assertTrue(TorchScriptDataType.FLOAT32.isSupported());
        assertTrue(TorchScriptDataType.FLOAT16.isSupported());
        assertTrue(TorchScriptDataType.INT64.isSupported());
        assertTrue(TorchScriptDataType.BOOL.isSupported());
        assertFalse(TorchScriptDataType.COMPLEX64.isSupported());
        assertFalse(TorchScriptDataType.COMPLEX128.isSupported());
    }

    @Test
    @DisplayName("Test isFloatingPoint")
    void testIsFloatingPoint() {
        assertTrue(TorchScriptDataType.FLOAT32.isFloatingPoint());
        assertTrue(TorchScriptDataType.FLOAT16.isFloatingPoint());
        assertTrue(TorchScriptDataType.BFLOAT16.isFloatingPoint());
        assertTrue(TorchScriptDataType.FLOAT64.isFloatingPoint());
        assertTrue(TorchScriptDataType.COMPLEX64.isFloatingPoint());
        assertTrue(TorchScriptDataType.COMPLEX128.isFloatingPoint());
        assertFalse(TorchScriptDataType.INT32.isFloatingPoint());
        assertFalse(TorchScriptDataType.INT64.isFloatingPoint());
        assertFalse(TorchScriptDataType.BOOL.isFloatingPoint());
    }

    @Test
    @DisplayName("Test isInteger")
    void testIsInteger() {
        assertTrue(TorchScriptDataType.INT8.isInteger());
        assertTrue(TorchScriptDataType.INT16.isInteger());
        assertTrue(TorchScriptDataType.INT32.isInteger());
        assertTrue(TorchScriptDataType.INT64.isInteger());
        assertTrue(TorchScriptDataType.UINT8.isInteger());
        assertFalse(TorchScriptDataType.FLOAT32.isInteger());
        assertFalse(TorchScriptDataType.FLOAT16.isInteger());
        assertFalse(TorchScriptDataType.BOOL.isInteger());
    }

    @Test
    @DisplayName("Test bytes per element")
    void testBytesPerElement() {
        assertEquals(4, TorchScriptDataType.FLOAT32.getBytesPerElement());
        assertEquals(2, TorchScriptDataType.FLOAT16.getBytesPerElement());
        assertEquals(2, TorchScriptDataType.BFLOAT16.getBytesPerElement());
        assertEquals(8, TorchScriptDataType.FLOAT64.getBytesPerElement());
        assertEquals(1, TorchScriptDataType.INT8.getBytesPerElement());
        assertEquals(2, TorchScriptDataType.INT16.getBytesPerElement());
        assertEquals(4, TorchScriptDataType.INT32.getBytesPerElement());
        assertEquals(8, TorchScriptDataType.INT64.getBytesPerElement());
        assertEquals(1, TorchScriptDataType.UINT8.getBytesPerElement());
        assertEquals(1, TorchScriptDataType.BOOL.getBytesPerElement());
    }

    @Test
    @DisplayName("Test torch name getter")
    void testTorchNameGetter() {
        assertEquals("torch.float32", TorchScriptDataType.FLOAT32.getTorchName());
        assertEquals("torch.float16", TorchScriptDataType.FLOAT16.getTorchName());
        assertEquals("torch.int64", TorchScriptDataType.INT64.getTorchName());
    }

    @Test
    @DisplayName("Test safetensors name getter")
    void testSafetensorsNameGetter() {
        assertEquals("F32", TorchScriptDataType.FLOAT32.getSafetensorsName());
        assertEquals("F16", TorchScriptDataType.FLOAT16.getSafetensorsName());
        assertEquals("BF16", TorchScriptDataType.BFLOAT16.getSafetensorsName());
        assertEquals("I64", TorchScriptDataType.INT64.getSafetensorsName());
    }
}
