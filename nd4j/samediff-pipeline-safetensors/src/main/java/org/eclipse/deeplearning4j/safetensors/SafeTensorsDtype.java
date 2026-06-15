/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.safetensors;

import org.nd4j.linalg.api.buffer.DataType;

/**
 * SafeTensors data type mappings to ND4J DataType.
 */
public enum SafeTensorsDtype {
    F64("F64", DataType.DOUBLE, 8),
    F32("F32", DataType.FLOAT, 4),
    F16("F16", DataType.HALF, 2),
    BF16("BF16", DataType.BFLOAT16, 2),
    I64("I64", DataType.INT64, 8),
    I32("I32", DataType.INT32, 4),
    I16("I16", DataType.INT16, 2),
    I8("I8", DataType.INT8, 1),
    U8("U8", DataType.UINT8, 1),
    U16("U16", DataType.UINT16, 2),
    U32("U32", DataType.UINT32, 4),
    U64("U64", DataType.UINT64, 8),
    BOOL("BOOL", DataType.BOOL, 1);

    private final String name;
    private final DataType nd4jType;
    private final int bytesPerElement;

    SafeTensorsDtype(String name, DataType nd4jType, int bytesPerElement) {
        this.name = name;
        this.nd4jType = nd4jType;
        this.bytesPerElement = bytesPerElement;
    }

    public String getName() {
        return name;
    }

    public DataType toNd4jType() {
        return nd4jType;
    }

    public int getBytesPerElement() {
        return bytesPerElement;
    }

    public int getElementSize() {
        return bytesPerElement;
    }

    public static SafeTensorsDtype fromNd4jType(DataType dataType) {
        for (SafeTensorsDtype d : values()) {
            if (d.nd4jType == dataType) {
                return d;
            }
        }
        throw new IllegalArgumentException("No SafeTensors dtype mapping for ND4J type: " + dataType);
    }

    public static SafeTensorsDtype fromString(String dtype) {
        for (SafeTensorsDtype d : values()) {
            if (d.name.equalsIgnoreCase(dtype)) {
                return d;
            }
        }
        throw new IllegalArgumentException("Unknown SafeTensors dtype: " + dtype);
    }

    public static DataType toNd4jDataType(String dtype) {
        return fromString(dtype).toNd4jType();
    }
}
