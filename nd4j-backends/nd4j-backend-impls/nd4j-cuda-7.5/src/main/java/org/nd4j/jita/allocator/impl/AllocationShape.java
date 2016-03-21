package org.nd4j.jita.allocator.impl;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class AllocationShape {
    private int offset = 0;
    private int length = 0;
    private int stride = 1;
    private int elementSize = 0;
    private DataBuffer.Type dataType = DataBuffer.Type.FLOAT;

    public AllocationShape(int length, int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
    }
}
