package org.nd4j.linalg.jcublas.buffer.allocation;

import jcuda.Pointer;
import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Host and device pointers.
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class HostDevicePointer {
    private Pointer hostPointer;
    private Pointer devicePointer;
}
