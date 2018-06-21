package org.nd4j.linalg.api.shape.options;

import lombok.val;
import org.nd4j.linalg.api.shape.Shape;

public class ArrayOptionsHelper {
    public static boolean hasBitSet(long[] shapeInfo, long bit) {
        val opt = Shape.options(shapeInfo);

        return hasBitSet(opt, bit);
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


    public static void setOptionBit(long[] storage, ArrayType type) {
        int length = Shape.shapeInfoLength(storage);
        storage[length - 3] = setOptionBit(storage[length-3], type);
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

}
