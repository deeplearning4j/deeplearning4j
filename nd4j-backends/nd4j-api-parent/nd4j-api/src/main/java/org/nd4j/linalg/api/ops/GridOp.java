package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ops.grid.GridDescriptor;

/**
 * MetaOp is special op, that contains multiple ops
 *
 * @author raver119@gmail.com
 */
public interface GridOp extends Op {

    /**
     *
     * @return
     */
    GridDescriptor getGridDescriptor();
}
