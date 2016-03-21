package org.nd4j.jita.allocator.pointers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PointersPair {
    /**
     * this field can be null, on system without any special devices
     */
    private Pointer devicePointer;

    /**
     * this should always contain long pointer to host memory
     */
    private Pointer hostPointer;

    public PointersPair(long devicePointer, long hostPointer) {
        this.devicePointer = new CudaPointer(devicePointer);
        this.hostPointer = new CudaPointer(hostPointer);
    }
}
