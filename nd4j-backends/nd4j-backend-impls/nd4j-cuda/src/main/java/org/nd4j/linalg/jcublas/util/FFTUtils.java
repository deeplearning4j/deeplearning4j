package org.nd4j.linalg.jcublas.util;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author Adam Gibson
 */
public class FFTUtils {
    /**
     * Get the plan for the given buffer (C2C for float Z2Z for double)
     * @param buff the buffer to get the plan for
     * @return the plan for the given buffer
     */
    public static int getPlanFor(DataBuffer buff) {
        /*   if(buff.dataType() == DataBuffer.Type.FLOAT)
            return cufftType.CUFFT_C2C;
        else
            return cufftType.CUFFT_Z2Z;
            */
        throw new UnsupportedOperationException();
    }


}
