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

package org.nd4j.linalg.api.shape.options;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

public class ArrayOptionsHelper {
    public static boolean hasBitSet(long[] shapeInfo, long bit) {
        val opt = Shape.options(shapeInfo);

        return hasBitSet(opt, bit);
    }

    public static void setOptionBit(long[] storage, ArrayType type) {
        int length = Shape.shapeInfoLength(storage);
        storage[length - 3] = setOptionBit(storage[length-3], type);
    }

    public static boolean hasBitSet(long storage, long bit) {
        return ((storage & bit) == bit);
    }

    public static ArrayType arrayType(long[] shapeInfo) {
        val opt = Shape.options(shapeInfo);

        if (hasBitSet(opt, 2))
            return ArrayType.SPARSE;
        else if (hasBitSet(opt, 4))
            return ArrayType.COMPRESSED;
        else if (hasBitSet(opt, 8))
            return ArrayType.EMPTY;
        else
            return ArrayType.DENSE;
    }

    public static DataType dataType(long opt) {
        if (hasBitSet(opt, 4))
            return DataType.COMPRESSED;
        else if (hasBitSet(opt, 4096))
            return DataType.HALF;
        else if (hasBitSet(opt, 8192))
            return DataType.FLOAT;
        else if (hasBitSet(opt, 16384))
            return DataType.DOUBLE;
        else if (hasBitSet(opt, 131072))
            return DataType.INT;
        else if (hasBitSet(opt, 262144))
            return DataType.LONG;
        else if (hasBitSet(opt, 524288))
            return DataType.BOOL;
        else if (hasBitSet(opt, 32768))
            return DataType.BYTE;
        else if (hasBitSet(opt, 65536))
            return DataType.SHORT;
        else if (hasBitSet(opt, 1048576))
            return DataType.UTF8;
        else
            throw new ND4JIllegalStateException("Unknown extras set: [" + opt + "]");
    }

    public static DataType dataType(long[] shapeInfo) {
        val opt = Shape.options(shapeInfo);
        return dataType(opt);
    }

    public static long setOptionBit(long storage, DataType type) {
        long bit = 0;
        switch (type) {
            case HALF:
                bit = 4096;
                break;
            case FLOAT:
                bit = 8192;
                break;
            case DOUBLE:
                bit = 16384;
                break;
            case INT:
                bit = 131072;
                break;
            case LONG:
                bit = 262144;
                break;
            case BOOL:
                bit = 524288;
                break;
            case BYTE:
                bit = 32768;
                break;
            case SHORT:
                bit = 65536;
                break;
            case UTF8:
                bit = 1048576;
                break;
            case COMPRESSED:
                bit = 4;
                break;
            case UNKNOWN:
                throw new UnsupportedOperationException();
        }

        storage |= bit;
        return storage;
    }

    public static long setOptionBit(long storage, ArrayType type) {
        long bit = 0;
        switch (type) {
            case SPARSE:
                bit = 2L;
                break;
            case COMPRESSED:
                bit = 4L;
                break;
            case EMPTY:
                bit = 8L;
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
            case DT_BOOL:
                return DataType.BOOL;
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
            default:
                throw new ND4JIllegalStateException("Unknown data type used: [" + dataType + "]");
        }
    }

}
