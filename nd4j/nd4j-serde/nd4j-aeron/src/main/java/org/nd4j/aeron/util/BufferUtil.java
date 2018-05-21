package org.nd4j.aeron.util;


import java.nio.ByteBuffer;

/**
 * Minor {@link ByteBuffer} utils
 *
 * @author Adam Gibson
 */
public class BufferUtil {
    /**
     * Merge all byte buffers together
     * @param buffers the bytebuffers to merge
     * @param overAllCapacity the capacity of the
     *                        merged bytebuffer
     * @return the merged byte buffer
     *
     */
    public static ByteBuffer concat(ByteBuffer[] buffers, int overAllCapacity) {
        ByteBuffer all = ByteBuffer.allocateDirect(overAllCapacity);
        for (int i = 0; i < buffers.length; i++) {
            ByteBuffer curr = buffers[i].slice();
            all.put(curr);
        }

        all.rewind();
        return all;
    }

    /**
     * Merge all bytebuffers together
     * @param buffers the bytebuffers to merge
     * @return the merged bytebuffer
     */
    public static ByteBuffer concat(ByteBuffer[] buffers) {
        int overAllCapacity = 0;
        for (int i = 0; i < buffers.length; i++)
            overAllCapacity += buffers[i].limit() - buffers[i].position();
        //padding
        overAllCapacity += buffers[0].limit() - buffers[0].position();
        ByteBuffer all = ByteBuffer.allocateDirect(overAllCapacity);
        for (int i = 0; i < buffers.length; i++) {
            ByteBuffer curr = buffers[i];
            all.put(curr);
        }

        all.flip();
        return all;
    }

}
