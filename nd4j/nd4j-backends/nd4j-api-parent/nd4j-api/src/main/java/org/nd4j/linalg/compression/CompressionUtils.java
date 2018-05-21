package org.nd4j.linalg.compression;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * This class provides utility methods for Compression in ND4J
 *
 * @author raver119@gmail.com
 */
public class CompressionUtils {

    public static boolean goingToDecompress(@NonNull DataBuffer.TypeEx from, @NonNull DataBuffer.TypeEx to) {
        // TODO: eventually we want FLOAT16 here
        if (to.equals(DataBuffer.TypeEx.FLOAT) || to.equals(DataBuffer.TypeEx.DOUBLE) )
            return true;

        return false;
    }

    public static boolean goingToCompress(@NonNull DataBuffer.TypeEx from, @NonNull DataBuffer.TypeEx to) {
        if (!goingToDecompress(from, to) && goingToDecompress(to, from))
            return true;

        return false;
    }
}
