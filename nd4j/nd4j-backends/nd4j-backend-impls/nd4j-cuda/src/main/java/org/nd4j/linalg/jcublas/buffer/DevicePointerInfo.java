package org.nd4j.linalg.jcublas.buffer;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.jita.allocator.pointers.PointersPair;

/**
 * Provides information about a device pointer
 *
 * @author bam4d
 */
@Data
@AllArgsConstructor
public class DevicePointerInfo {
    final private PointersPair pointers;
    final private long length;
    final private int stride;
    final private int offset;
    private boolean freed = false;


}
