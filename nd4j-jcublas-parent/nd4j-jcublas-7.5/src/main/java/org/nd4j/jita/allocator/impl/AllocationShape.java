package org.nd4j.jita.allocator.impl;

import lombok.Data;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
@Data
public class AllocationShape {
    private long offset = 0;
    private long length = 0;
    private long stride = 1;
    private DataBuffer.Type dataType = DataBuffer.Type.FLOAT;
}
