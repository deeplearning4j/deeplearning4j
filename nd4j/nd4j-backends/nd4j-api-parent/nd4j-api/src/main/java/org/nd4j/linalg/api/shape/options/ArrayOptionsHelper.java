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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;

import java.util.ArrayList;
import java.util.List;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT16;
import static org.nd4j.linalg.api.buffer.DataType.INT32;

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

    public static final long ARRAY_NEEDS_COPY = 67108864;


    //this is used for when we need to use the offset of an input
    //when creating a view of an array
    //see ArrayOptions.h for more information
    public static final long ARRAY_COPY_OFFSET_INPUT_0 = 134217728;
    public static final long ARRAY_COPY_OFFSET_INPUT_1 = 268435456;
    public static final long ARRAY_COPY_OFFSET_INPUT_2 = 536870912;
    public static final long ARRAY_COPY_OFFSET_INPUT_3 = 1073741824;
    public static final long ARRAY_COPY_OFFSET_INPUT_4 = 2147483648L;
    public static final long ARRAY_COPY_OFFSET_INPUT_5 = 4294967296L;
    public static final long ARRAY_COPY_OFFSET_INPUT_6 = 8589934592L;
    public static final long ARRAY_COPY_OFFSET_INPUT_7 = 17179869184L;
    public static final long ARRAY_COPY_OFFSET_INPUT_8 = 34359738368L;
    public static final long ARRAY_COPY_OFFSET_INPUT_9 = 68719476736L;
    public static final long ARRAY_COPY_OFFSET_INPUT_10 = 137438953472L;


    public static final long[] ARRAY_COPY_OFFSET_INDEXES = {
            ARRAY_COPY_OFFSET_INPUT_0,
            ARRAY_COPY_OFFSET_INPUT_1,
            ARRAY_COPY_OFFSET_INPUT_2,
            ARRAY_COPY_OFFSET_INPUT_3,
            ARRAY_COPY_OFFSET_INPUT_4,
            ARRAY_COPY_OFFSET_INPUT_5,
            ARRAY_COPY_OFFSET_INPUT_6,
            ARRAY_COPY_OFFSET_INPUT_7,
            ARRAY_COPY_OFFSET_INPUT_8,
            ARRAY_COPY_OFFSET_INPUT_9,
            ARRAY_COPY_OFFSET_INPUT_10
    };



    /**
     * Perform typical checks and compose them into a single flag.
     * This one just sets the data types and composes the other ones
     * all as false.
     * @param dataType the data type.
     * @return
     */
    public static long composeTypicalChecks(DataType dataType) {
        return composeTypicalChecks(false, dataType, false, false, false, false, false);
    }

    /**
     * Perform typical checks and compose them into a single flag.
     *
     * @param isEmpty Whether the array is empty.
     * @param dataType The data type of the array.
     * @param isSparse Whether the array is sparse.
     * @param isCompressed Whether the array is compressed.
     * @param isView Whether the array is a view.
     * @param needsCopy Whether the array needs to be copied.
     * @param hasPaddedBuffer Whether the array has a padded buffer.
     * @return A long value representing the composed options.
     */
    public static long composeTypicalChecks(boolean isEmpty, DataType dataType, boolean isSparse,
                                            boolean isCompressed, boolean isView, boolean needsCopy,
                                            boolean hasPaddedBuffer) {
        List<Long> flags = new ArrayList<>();

        if (isEmpty) {
            flags.add(ATYPE_EMPTY_BIT);
        }

        if (isSparse) {
            flags.add(ATYPE_SPARSE_BIT);
        }

        if (isCompressed) {
            flags.add(ATYPE_COMPRESSED_BIT);
        }

        // Add data type flag
        if (dataType == DataType.COMPRESSED) {
            flags.add(DTYPE_COMPRESSED_BIT);
        } else if (dataType == DataType.FLOAT16) {
            flags.add(DTYPE_HALF_BIT);
        } else if (dataType == DataType.BFLOAT16) {
            flags.add(DTYPE_BFLOAT16_BIT);
        } else if (dataType == DataType.FLOAT) {
            flags.add(DTYPE_FLOAT_BIT);
        } else if (dataType == DataType.DOUBLE) {
            flags.add(DTYPE_DOUBLE_BIT);
        } else if (dataType == DataType.INT32) {
            flags.add(DTYPE_INT_BIT);
        } else if (dataType == DataType.INT64) {
            flags.add(DTYPE_LONG_BIT);
        } else if (dataType == DataType.BOOL) {
            flags.add(DTYPE_BOOL_BIT);
        } else if (dataType == DataType.INT8) {
            flags.add(DTYPE_BYTE_BIT);
        } else if (dataType == DataType.INT16) {
            flags.add(DTYPE_SHORT_BIT);
        } else if (dataType == DataType.UTF8) {
            flags.add(DTYPE_UTF8_BIT);
        } else if (dataType == DataType.UINT8 || dataType == DataType.UINT16 ||
                dataType == DataType.UINT32 || dataType == DataType.UINT64) {
            flags.add(DTYPE_UNSIGNED_BIT);
            // Also add the corresponding signed type bit
            if (dataType == DataType.UINT8) flags.add(DTYPE_BYTE_BIT);
            else if (dataType == DataType.UINT16) flags.add(DTYPE_SHORT_BIT);
            else if (dataType == DataType.UINT32) flags.add(DTYPE_INT_BIT);
            else if (dataType == DataType.UINT64) flags.add(DTYPE_LONG_BIT);
        } else {
            throw new IllegalArgumentException("Unknown DataType: " + dataType);
        }

        if (isView) {
            flags.add(IS_VIEW);
        }

        if (needsCopy) {
            flags.add(ARRAY_NEEDS_COPY);
        }

        if (hasPaddedBuffer) {
            flags.add(HAS_PADDED_BUFFER);
        }

        return composeOptions(flags);
    }

    /**
     * Compose a single option value from a list of flags and a base option value.
     *
     * @param flags A list of long flags to be composed into the option value.
     * @param baseOption The base option value to start with. Defaults to 0 if not provided.
     * @return A long value representing the composed options.
     */
    public static long composeOptions(List<Long> flags, long baseOption) {
        long composedOption = baseOption;

        for (Long flag : flags) {
            composedOption |= flag;
        }

        return composedOption;
    }

    /**
     * Overloaded method that uses a default base option of 0.
     *
     * @param flags A list of long flags to be composed into the option value.
     * @return A long value representing the composed options.
     */
    public static long composeOptions(List<Long> flags) {
        return composeOptions(flags, 0L);
    }

    /**
     * Returns true if an array needs a copy
     * after the output shape is calculated.
     * This is mainly meant for the c++ usage of
     * {@link Shape#newShapeNoCopy(INDArray, int[], boolean)}
     * that normally returns null. This flag can be used in other ways
     * as well for anything to do with views.
     * @param shapeInfo the shape info to check
     * @return
     */
    public static boolean arrayNeedsCopy(long[] shapeInfo) {
        return hasBitSet(shapeInfo, ARRAY_NEEDS_COPY);
    }

    /**
     * Returns true if an array needs a copy
     * after the output shape is calculated.
     * This is mainly meant for the c++ usage of
     * {@link Shape#newShapeNoCopy(INDArray, int[], boolean)}
     * that normally returns null. This flag can be used in other ways
     * as well for anything to do with views.
     * @param shapeInfo the shape info to check
     * @return
     */
    public static boolean arrayNeedsCopy(long shapeInfo) {
        return hasBitSet(shapeInfo, ARRAY_NEEDS_COPY);
    }

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
     * Returns true if the given shape info has the
     * {@link #hasBitSet(long, long)} with the property
     * {@link #ATYPE_EMPTY_BIT}
     * @param shapeInfo the shape info to check
     * @return
     */
    public static boolean isEmpty(long shapeInfo) {
        return hasBitSet(shapeInfo, ATYPE_EMPTY_BIT);
    }



    /**
     * Returns true if the given shape info has the
     * {@link #hasBitSet(long, long)} with the property
     * {@link #ATYPE_EMPTY_BIT}
     * @param shapeInfo the shape info to check
     * @return
     */
    public static boolean isEmpty(long[] shapeInfo) {
        return hasBitSet(shapeInfo, ATYPE_EMPTY_BIT);
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
     * Set the data type in the shape info array
     * @param shapeInfo the shape info array to modify
     * @param dataType the DataType to set
     */
    public static void setDataTypeInShapeInfo(long[] shapeInfo, DataType dataType) {
        long options = getOptions(shapeInfo);
        long updatedOptions = setDataType(options, dataType);
        setOptions(shapeInfo, updatedOptions);
    }

    /**
     * Get the options from the shape info array
     * @param buffer the shape info array
     * @return the options value
     */
    private static long getOptions(long[] buffer) {
        long rank = Shape.rank(buffer);
        int idx = rank == 0 ? 3 : (int) (rank + rank + 1);
        return buffer[idx];
    }

    /**
     * Set the options in the shape info array
     * @param buffer the shape info array
     * @param options the options value to set
     */
    private static void setOptions(long[] buffer, long options) {
        long rank = Shape.rank(buffer);
        int idx = rank == 0 ? 3 : (int) (rank + rank + 1);
        buffer[idx] = options;
    }


    /**
     * Set the appropriate bits for the given {@link DataType}
     * @param opt the existing options value to modify
     * @param dataType the DataType to set the bits for
     * @return the modified options value with the appropriate bits set for the given DataType
     */
    public static long setDataType(long opt, DataType dataType) {
        // Clear existing data type bits
        opt &= ~((1L << DTYPE_COMPRESSED_BIT) | (1L << DTYPE_HALF_BIT) | (1L << DTYPE_BFLOAT16_BIT) |
                (1L << DTYPE_FLOAT_BIT) | (1L << DTYPE_DOUBLE_BIT) | (1L << DTYPE_INT_BIT) |
                (1L << DTYPE_LONG_BIT) | (1L << DTYPE_BOOL_BIT) | (1L << DTYPE_BYTE_BIT) |
                (1L << DTYPE_SHORT_BIT) | (1L << DTYPE_UTF8_BIT) | (1L << DTYPE_UNSIGNED_BIT));

        // Set the appropriate bits based on the DataType
        if (dataType == DataType.COMPRESSED) {
            opt |= 1L << DTYPE_COMPRESSED_BIT;
        } else if (dataType == DataType.FLOAT16) {
            opt |= 1L << DTYPE_HALF_BIT;
        } else if (dataType == DataType.BFLOAT16) {
            opt |= 1L << DTYPE_BFLOAT16_BIT;
        } else if (dataType == DataType.FLOAT) {
            opt |= 1L << DTYPE_FLOAT_BIT;
        } else if (dataType == DataType.DOUBLE) {
            opt |= 1L << DTYPE_DOUBLE_BIT;
        } else if (dataType == DataType.UINT32) {
            opt |= 1L << DTYPE_INT_BIT;
            opt |= 1L << DTYPE_UNSIGNED_BIT;
        } else if (dataType == INT32) {
            opt |= 1L << DTYPE_INT_BIT;
        } else if (dataType == DataType.UINT64) {
            opt |= 1L << DTYPE_LONG_BIT;
            opt |= 1L << DTYPE_UNSIGNED_BIT;
        } else if (dataType == DataType.INT64) {
            opt |= 1L << DTYPE_LONG_BIT;
        } else if (dataType == DataType.BOOL) {
            opt |= 1L << DTYPE_BOOL_BIT;
        } else if (dataType == DataType.UINT8) {
            opt |= 1L << DTYPE_BYTE_BIT;
            opt |= 1L << DTYPE_UNSIGNED_BIT;
        } else if (dataType == DataType.INT8) {
            opt |= 1L << DTYPE_BYTE_BIT;
        } else if (dataType == DataType.UINT16) {
            opt |= 1L << DTYPE_SHORT_BIT;
            opt |= 1L << DTYPE_UNSIGNED_BIT;
        } else if (dataType == DataType.INT16) {
            opt |= 1L << DTYPE_SHORT_BIT;
        } else if (dataType == DataType.UTF8) {
            opt |= 1L << DTYPE_UTF8_BIT;
        } else {
            throw new IllegalArgumentException("Unknown DataType: " + dataType);
        }

        return opt;
    }



    public static DataType dataType(long opt) {
        if (hasBitSet(opt, DTYPE_COMPRESSED_BIT))
            return DataType.COMPRESSED;
        else if (hasBitSet(opt, DTYPE_HALF_BIT))
            return FLOAT16;
        else if (hasBitSet(opt, DTYPE_BFLOAT16_BIT))
            return DataType.BFLOAT16;
        else if (hasBitSet(opt, DTYPE_FLOAT_BIT))
            return DataType.FLOAT;
        else if (hasBitSet(opt, DTYPE_DOUBLE_BIT))
            return DataType.DOUBLE;
        else if (hasBitSet(opt, DTYPE_INT_BIT))
            return hasBitSet(opt, DTYPE_UNSIGNED_BIT) ? DataType.UINT32 : INT32;
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
