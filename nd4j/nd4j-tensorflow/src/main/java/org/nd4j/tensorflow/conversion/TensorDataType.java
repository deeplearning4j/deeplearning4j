/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.tensorflow.conversion;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;

public enum TensorDataType {
    INVALID,
    FLOAT,
    DOUBLE,
    INT32,
    UINT8,
    INT16,
    INT8,
    STRING,
    COMPLEX64,
    INT64,
    BOOL,
    QINT8,
    QUINT8,
    QINT32,
    BFLOAT16,
    QINT16,
    QUINT16,
    UINT16,
    COMPLEX128,
    HALF,
    RESOURCE,
    VARIANT,
    UINT32,
    UINT64;


    /**
     * Map a tensor data type to a proto value found in tensorflow.
     * Generally, this is just replacing DT_ with empty
     * and returning enum.valueOf(string)
     * @param value the input string
     * @return the associated {@link TensorDataType}
     */
    public static TensorDataType fromProtoValue(String value) {
        String valueReplace = value.replace("DT_","");
        return TensorDataType.valueOf(valueReplace);
    }



    /**
     * Get the python name for the given data type
     * @param tensorDataType the python name for the given data type
     * @return float64 for double, float32 for double, float16 for half, otherwise
     * the type's name converted to lower case
     */
    public static String toPythonName(TensorDataType tensorDataType) {
        switch(tensorDataType) {
            case DOUBLE: return "float64";
            case FLOAT: return "float32";
            case HALF: return "float16";

            default: return tensorDataType.name().toLowerCase();
        }
    }

    public static DataType toNd4jType(TensorDataType tensorDataType) {
        switch(tensorDataType) {
            case FLOAT: return DataType.FLOAT;
            case DOUBLE: return DataType.DOUBLE;
            case BOOL: return DataType.BOOL;
            case INT32: return DataType.INT;
            case INT64: return DataType.LONG;
            case STRING: return DataType.UTF8;
            case HALF: return DataType.HALF;
            default: throw new IllegalArgumentException("Unsupported type " + tensorDataType.name());
        }
    }


    public static TensorDataType fromNd4jType(DataType dataType) {
        switch(dataType) {
            case FLOAT: return TensorDataType.FLOAT;
            case LONG: return TensorDataType.INT64;
            case INT: return TensorDataType.INT32;
            case BOOL: return TensorDataType.BOOL;
            case DOUBLE: return TensorDataType.DOUBLE;
            case HALF: return TensorDataType.HALF;
            case UTF8: return TensorDataType.STRING;
            case COMPRESSED: throw new IllegalStateException("Unable to work with compressed data type. Could be 1 or more types.");
            case SHORT: return TensorDataType.INT16;
            default: throw new IllegalArgumentException("Unknown data type " + dataType);
        }
    }

    public static TensorDataType fromNd4jType(INDArray array) {
        DataType dataType = array.dataType();
        switch(dataType) {
            case COMPRESSED:
                CompressedDataBuffer compressedData = (CompressedDataBuffer) array.data();
                CompressionDescriptor desc = compressedData.getCompressionDescriptor();
                String algo = desc.getCompressionAlgorithm();
                switch (algo) {
                    case "FLOAT16": return HALF;
                    case "INT8":    return INT8;
                    case "UINT8":  return UINT8;
                    case "INT16":   return INT16;
                    case "UINT16":  return UINT16;
                    default: throw new IllegalArgumentException("Unsupported compression algorithm: " + algo);
                }

            default: return fromNd4jType(dataType);
        }
    }

}
