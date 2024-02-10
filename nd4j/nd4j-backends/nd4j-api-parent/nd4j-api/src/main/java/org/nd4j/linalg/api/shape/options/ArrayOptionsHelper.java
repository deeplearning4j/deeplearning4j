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

package org.nd4j.linalg.api.shape.options;

import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;

/**
 * This is a mirror of the C++ ArrayOptionsHelper class.
 * It contains the bit mask options for the shape information buffer.
 * These options are used to determine the data type, array type, and other
 * properties of the array.
 *
 * The ArrayOptions implementation and header for c++ can be found at
 * this link:
 * https://github.com/deeplearning4j/deeplearning4j/blob/master/libnd4j/include/array/ArrayOptions.h
 *
 *
 *
 */
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

    public static final long IS_VIEW = 33554432;

    /**
     * Returns true if the given shape info has the
     * {@link #hasBitSet(long, long)} with the property
     * {@link #IS_VIEW}
     * @param shapeInfo the shape info to check
     * @return
     */
    public static boolean isView(long shapeInfo) {
        return hasBitSet(shapeInfo, IS_VIEW);
    }



    /**
     * Returns true if the given shape info has the
     * {@link #hasBitSet(long, long)} with the property
     * {@link #IS_VIEW}
     * @param shapeInfo the shape info to check
     * @return
     */
    public static boolean isView(long[] shapeInfo) {
        return hasBitSet(shapeInfo, IS_VIEW);
    }

    /**
     * Toggle whether the the given bit is set
     * @param flagStorage the storage to toggle
     * @param property the property to toggle
     * @return the new property value
     */
    public static long toggleBitSet(long flagStorage,long property) {
        return flagStorage ^= property;
    }

    /**
     * Toggle whether the array is a view or not
     * @param property the property to toggle
     * @return the new property value
     */
    public static long toggleHasView(long property) {
        return toggleBitSet(property, IS_VIEW);
    }

    /**
     * Toggle whether the array has a padded buffer or not
     * @param bit the property to toggle
     * @return
     */
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


    public static ArrayType arrayType(long opt) {
        if (hasBitSet(opt, ATYPE_SPARSE_BIT))
            return ArrayType.SPARSE;
        else if (hasBitSet(opt, ATYPE_COMPRESSED_BIT))
            return ArrayType.COMPRESSED;
        else if (hasBitSet(opt, ATYPE_EMPTY_BIT))
            return ArrayType.EMPTY;
        else
            return ArrayType.DENSE;
    }

    /**
     * Return the {@link ArrayType} for the given shape info buffer
     * @param shapeInfo the shape info buffer to get the array type for
     * @return the array type for the given shape info buffer
     */
    public static ArrayType arrayType(long[] shapeInfo) {
        return arrayType(Shape.options(shapeInfo));
    }

    /**
     * Return the {@link DataType} for the given shape info buffer
     * @param opt the long storage to get the data type for
     * @return the data type for the given shape info buffer
     */
    public static DataType dataType(long opt) {
        if (hasBitSet(opt, DTYPE_COMPRESSED_BIT))
            return DataType.COMPRESSED;
        else if (hasBitSet(opt, DTYPE_HALF_BIT))
            return DataType.FLOAT16;
        else if (hasBitSet(opt, DTYPE_BFLOAT16_BIT))
            return DataType.BFLOAT16;
        else if (hasBitSet(opt, DTYPE_FLOAT_BIT))
            return DataType.FLOAT;
        else if (hasBitSet(opt, DTYPE_DOUBLE_BIT))
            return DataType.DOUBLE;
        else if (hasBitSet(opt, DTYPE_INT_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT32 : DataType.INT32;
        else if (hasBitSet(opt, DTYPE_LONG_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT64 : DataType.INT64;
        else if (hasBitSet(opt, DTYPE_BOOL_BIT))
            return DataType.BOOL;
        else if (hasBitSet(opt, DTYPE_BYTE_BIT)) {
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT8 : DataType.INT8;     //Byte bit set for both UBYTE and BYTE
        } else if (hasBitSet(opt, DTYPE_SHORT_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT16 : DataType.INT16;
        else if (hasBitSet(opt, DTYPE_UTF8_BIT))
            return DataType.UTF8;
        else
            throw new ND4JUnknownDataTypeException("Unknown extras set: [" + opt + "]");
    }

    /**
     * Return the data type for the given shape info buffer
     * @param shapeInfo the shape info buffer to get the data type for
     * @return the data type for the given shape info buffer
     */
    public static DataType dataType(long[] shapeInfo) {
        val opt = Shape.options(shapeInfo);
        return dataType(opt);
    }

    /**
     * Return the data type for the given shape info buffer
     * @param storage the storage value to set the bit for
     * @param type the data type to set the bit for
     * @return the data type for the given shape info buffer
     */
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

    /**
     * Set the option bit for the array type.
     * @param storage the storage value to set the bit for
     * @param type the array type to set the bit for
     * @return the new storage value with the bit set
     */
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

}
