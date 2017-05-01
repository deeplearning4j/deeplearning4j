package org.nd4j.linalg.api.memory.pointers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PointersPair {
    private Long allocationCycle;
    private Long requiredMemory;
    private PagedPointer hostPointer;
    private PagedPointer devicePointer;

    public PointersPair(PagedPointer hostPointer, PagedPointer devicePointer) {
        if (hostPointer == null && devicePointer == null)
            throw new RuntimeException("Both pointers can't be null");

        this.hostPointer = hostPointer;
        this.devicePointer = devicePointer;
    }
}
