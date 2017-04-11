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
    private PagedPointer hostPointer;
    private PagedPointer devicePointer;
}
