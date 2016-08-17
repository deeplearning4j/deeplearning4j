package org.nd4j.linalg.api.ops.grid;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class GridDescriptor {
    /**
     * Number of ops in grid
     */
    private int gridDepth;
}
