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

package org.nd4j.torchscript.format;

import lombok.Getter;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.HashMap;
import java.util.Map;

/**
 * PyTorch data types and their ND4J equivalents.
 */
@Getter
public enum TorchScriptDataType {
    FLOAT32("torch.float32", "F32", DataType.FLOAT, 4),
    FLOAT16("torch.float16", "F16", DataType.HALF, 2),
    BFLOAT16("torch.bfloat16", "BF16", DataType.BFLOAT16, 2),
    FLOAT64("torch.float64", "F64", DataType.DOUBLE, 8),
    INT8("torch.int8", "I8", DataType.INT8, 1),
    INT16("torch.int16", "I16", DataType.INT16, 2),
    INT32("torch.int32", "I32", DataType.INT32, 4),
    INT64("torch.int64", "I64", DataType.INT64, 8),
    UINT8("torch.uint8", "U8", DataType.UINT8, 1),
    BOOL("torch.bool", "BOOL", DataType.BOOL, 1),
    COMPLEX64("torch.complex64", null, null, 8),
    COMPLEX128("torch.complex128", null, null, 16);

    private final String torchName;
    private final String safetensorsName;
    private final DataType nd4jType;
    private final int bytesPerElement;

    private static final Map<String, TorchScriptDataType> TORCH_NAME_MAP = new HashMap<>();
    private static final Map<String, TorchScriptDataType> SAFETENSORS_MAP = new HashMap<>();

    static {
        for (TorchScriptDataType type : values()) {
            TORCH_NAME_MAP.put(type.torchName, type);
            // Also add short names (e.g., "float32" -> FLOAT32)
            String shortName = type.torchName.replace("torch.", "");
            TORCH_NAME_MAP.put(shortName, type);
            if (type.safetensorsName != null) {
                SAFETENSORS_MAP.put(type.safetensorsName, type);
            }
        }
        // Add additional PyTorch aliases
        TORCH_NAME_MAP.put("torch.float", FLOAT32);
        TORCH_NAME_MAP.put("torch.double", FLOAT64);
        TORCH_NAME_MAP.put("torch.half", FLOAT16);
        TORCH_NAME_MAP.put("torch.int", INT32);
        TORCH_NAME_MAP.put("torch.long", INT64);
        TORCH_NAME_MAP.put("torch.short", INT16);
        TORCH_NAME_MAP.put("torch.byte", UINT8);
        TORCH_NAME_MAP.put("float", FLOAT32);
        TORCH_NAME_MAP.put("double", FLOAT64);
        TORCH_NAME_MAP.put("half", FLOAT16);
        TORCH_NAME_MAP.put("int", INT32);
        TORCH_NAME_MAP.put("long", INT64);
        TORCH_NAME_MAP.put("short", INT16);
        TORCH_NAME_MAP.put("byte", UINT8);
    }

    TorchScriptDataType(String torchName, String safetensorsName, DataType nd4jType, int bytesPerElement) {
        this.torchName = torchName;
        this.safetensorsName = safetensorsName;
        this.nd4jType = nd4jType;
        this.bytesPerElement = bytesPerElement;
    }

    /**
     * Get TorchScriptDataType from PyTorch type name.
     *
     * @param name PyTorch type name (e.g., "torch.float32" or "float32")
     * @return the corresponding TorchScriptDataType, or null if not found
     */
    public static TorchScriptDataType fromTorchName(String name) {
        return TORCH_NAME_MAP.get(name);
    }

    /**
     * Get TorchScriptDataType from Safetensors dtype string.
     *
     * @param dtype Safetensors dtype (e.g., "F32", "F16")
     * @return the corresponding TorchScriptDataType, or null if not found
     */
    public static TorchScriptDataType fromSafetensorsType(String dtype) {
        return SAFETENSORS_MAP.get(dtype);
    }

    /**
     * Check if this data type is supported by ND4J.
     *
     * @return true if there is a direct ND4J mapping
     */
    public boolean isSupported() {
        return nd4jType != null;
    }

    /**
     * Check if this is a floating point type.
     *
     * @return true if floating point
     */
    public boolean isFloatingPoint() {
        return this == FLOAT32 || this == FLOAT16 || this == BFLOAT16 ||
               this == FLOAT64 || this == COMPLEX64 || this == COMPLEX128;
    }

    /**
     * Check if this is an integer type.
     *
     * @return true if integer
     */
    public boolean isInteger() {
        return this == INT8 || this == INT16 || this == INT32 ||
               this == INT64 || this == UINT8;
    }
}
