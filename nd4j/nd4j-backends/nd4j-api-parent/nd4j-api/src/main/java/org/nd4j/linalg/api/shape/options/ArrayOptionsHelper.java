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

package org.nd4j.linalg.api.shape.options;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;

public class ArrayOptionsHelper {
    public static final long ATYPE_SPARSE_BIT = 2;
    public static final long ATYPE_COMPRESSED_BIT = 4;
    public static final long ATYPE_EMPTY_BIT = 8;

    public static final long DTYPE_COMPRESSED_BIT = 4;
    public static final long DTYPE_BFLOAT16_BIT = 2048;
    public static final long DTYPE_HALF_BIT = 4096;
    public static final long DTYPE_FLOAT_BIT = 8192;
    public static final long DTYPE_DOUBLE_BIT = 16384;
    public static final long DTYPE_INT_BIT = 131072;
    public static final long DTYPE_LONG_BIT = 262144;
    public static final long DTYPE_BOOL_BIT = 524288;
    public static final long DTYPE_BYTE_BIT = 32768;        //Also used for UBYTE in conjunction with sign bit
    public static final long DTYPE_SHORT_BIT = 65536;
    public static final long DTYPE_UTF8_BIT = 1048576;
    public static final long DTYPE_UNSIGNED_BIT = 8388608;

    public static final long HAS_PADDED_BUFFER = (1<<25);

    public static boolean hasBitSet(long[] shapeInfo, long bit) {
        val opt = Shape.options(shapeInfo);

        return hasBitSet(opt, bit);
    }

    public static long setOptionBit(long extras, long bit) {
        return extras | bit;
    }

    public static void setOptionBit(long[] storage, ArrayType type) {
        int length = Shape.shapeInfoLength(storage);
        storage[length - 3] = setOptionBit(storage[length - 3], type);
    }

    public static boolean hasBitSet(long storage, long bit) {
        return ((storage & bit) == bit);
    }

    public static ArrayType arrayType(long[] shapeInfo) {
        val opt = Shape.options(shapeInfo);

        if (hasBitSet(opt, ATYPE_SPARSE_BIT))
            return ArrayType.SPARSE;
        else if (hasBitSet(opt, ATYPE_COMPRESSED_BIT))
            return ArrayType.COMPRESSED;
        else if (hasBitSet(opt, ATYPE_EMPTY_BIT))
            return ArrayType.EMPTY;
        else
            return ArrayType.DENSE;
    }

    public static DataType dataType(long opt) {
        if (hasBitSet(opt, DTYPE_COMPRESSED_BIT))
            return DataType.COMPRESSED;
        else if (hasBitSet(opt, DTYPE_HALF_BIT))
            return DataType.HALF;
        else if (hasBitSet(opt, DTYPE_BFLOAT16_BIT))
            return DataType.BFLOAT16;
        else if (hasBitSet(opt, DTYPE_FLOAT_BIT))
            return DataType.FLOAT;
        else if (hasBitSet(opt, DTYPE_DOUBLE_BIT))
            return DataType.DOUBLE;
        else if (hasBitSet(opt, DTYPE_INT_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT32 : DataType.INT;
        else if (hasBitSet(opt, DTYPE_LONG_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT64 : DataType.LONG;
        else if (hasBitSet(opt, DTYPE_BOOL_BIT))
            return DataType.BOOL;
        else if (hasBitSet(opt, DTYPE_BYTE_BIT)) {
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UBYTE : DataType.BYTE;     //Byte bit set for both UBYTE and BYTE
        } else if (hasBitSet(opt, DTYPE_SHORT_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT16 : DataType.SHORT;
        else if (hasBitSet(opt, DTYPE_UTF8_BIT))
            return DataType.UTF8;
        else
            throw new ND4JUnknownDataTypeException("Unknown extras set: [" + opt + "]");
    }

    public static DataType dataType(long[] shapeInfo) {
        val opt = Shape.options(shapeInfo);
        return dataType(opt);
    }

    public static long setOptionBit(long storage, DataType type) {
        long bit = 0;
        switch (type) {
            case HALF:
                bit = DTYPE_HALF_BIT;
                break;
            case BFLOAT16:
                bit = DTYPE_BFLOAT16_BIT;
                break;
            case FLOAT:
                bit = DTYPE_FLOAT_BIT;
                break;
            case DOUBLE:
                bit = DTYPE_DOUBLE_BIT;
                break;
            case UINT32:
                storage |= DTYPE_UNSIGNED_BIT;
            case INT:
                bit = DTYPE_INT_BIT;
                break;
            case UINT64:
                storage |= DTYPE_UNSIGNED_BIT;
            case LONG:
                bit = DTYPE_LONG_BIT;
                break;
            case BOOL:
                bit = DTYPE_BOOL_BIT;
                break;
            case UBYTE:
                storage |= DTYPE_UNSIGNED_BIT; // unsigned bit
                //Intentional fallthrough
            case BYTE:
                bit = DTYPE_BYTE_BIT;
                break;
            case UINT16:
                storage |= DTYPE_UNSIGNED_BIT;
            case SHORT:
                bit = DTYPE_SHORT_BIT;
                break;
            case UTF8:
                bit = DTYPE_UTF8_BIT;
                break;
            case COMPRESSED:
                bit = DTYPE_COMPRESSED_BIT;
                break;
            case UNKNOWN:
            default:
                throw new UnsupportedOperationException();
        }

        storage |= bit;
        return storage;
    }

    public static long setOptionBit(long storage, ArrayType type) {
        long bit = 0;
        switch (type) {
            case SPARSE:
                bit = ATYPE_SPARSE_BIT;
                break;
            case COMPRESSED:
                bit = ATYPE_COMPRESSED_BIT;
                break;
            case EMPTY:
                bit = ATYPE_EMPTY_BIT;
                break;
            default:
            case DENSE:
                return storage;
        }

        storage |= bit;
        return storage;
    }

    public static DataType convertToDataType(org.tensorflow.framework.DataType dataType) {
        switch (dataType) {
            case DT_UINT16:
                return DataType.UINT16;
            case DT_UINT32:
                return DataType.UINT32;
            case DT_UINT64:
                return DataType.UINT64;
            case DT_BOOL:
                return DataType.BOOL;
            case DT_BFLOAT16:
                return DataType.BFLOAT16;
            case DT_FLOAT:
                return DataType.FLOAT;
            case DT_INT32:
                return DataType.INT;
            case DT_INT64:
                return DataType.LONG;
            case DT_INT8:
                return DataType.BYTE;
            case DT_INT16:
                return DataType.SHORT;
            case DT_DOUBLE:
                return DataType.DOUBLE;
            case DT_UINT8:
                return DataType.UBYTE;
            case DT_HALF:
                return DataType.HALF;
            case DT_STRING:
                return DataType.UTF8;
            default:
                throw new UnsupportedOperationException("Unknown TF data type: [" + dataType.name() + "]");
        }
    }

    public static DataType dataType(@NonNull String dataType) {
        switch (dataType) {
            case "uint64":
                return DataType.UINT64;
            case "uint32":
                return DataType.UINT32;
            case "uint16":
                return DataType.UINT16;
            case "int64":
                return DataType.LONG;
            case "int32":
                return DataType.INT;
            case "int16":
                return DataType.SHORT;
            case "int8":
                return DataType.BYTE;
            case "bool":
                return DataType.BOOL;
            case "resource": //special case, nodes like Enter
            case "float32":
                return DataType.FLOAT;
            case "float64":
            case "double":
                return DataType.DOUBLE;
            case "string":
                return DataType.UTF8;
            case "uint8":
            case "ubyte":
                return DataType.UBYTE;
            case "bfloat16":
                return DataType.BFLOAT16;
            case "float16":
                return DataType.HALF;
            default:
                throw new ND4JIllegalStateException("Unknown data type used: [" + dataType + "]");
        }
    }

}
