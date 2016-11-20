package org.nd4j.rng.deallocator;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * Simple wrapper for state pointer, to avoid enqueue of non-initialized objects
 *
 * @author raver119@gmail.com
 */
@Data
@AllArgsConstructor
public class NativePack {
    private Long address;
    private Pointer statePointer;
}
