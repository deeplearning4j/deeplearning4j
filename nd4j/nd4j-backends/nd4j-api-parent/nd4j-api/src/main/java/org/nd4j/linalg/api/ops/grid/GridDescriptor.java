package org.nd4j.linalg.api.ops.grid;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

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
    private int gridDepth = 0;

    /**
     * Ordered list of pointers for this grid
     */
    private List<GridPointers> gridPointers = new ArrayList<>();


}
