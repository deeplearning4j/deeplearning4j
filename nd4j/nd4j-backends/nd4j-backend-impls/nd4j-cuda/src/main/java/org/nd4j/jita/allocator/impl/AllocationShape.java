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
    private long offset = 0;
    private long length = 0;
    private int stride = 1;
    private int elementSize = 0;
    private DataBuffer.Type dataType = DataBuffer.Type.FLOAT;

    /*
    public AllocationShape(long length, int elementSize) {
        this.length = length;
        this.elementSize = elementSize;
    }
    */
    public AllocationShape(long length, int elementSize, DataBuffer.Type dataType) {
        this.length = length;
        this.elementSize = elementSize;
        this.dataType = dataType;
    }


    public long getNumberOfBytes() {
        return this.length * this.elementSize;
    }
}
