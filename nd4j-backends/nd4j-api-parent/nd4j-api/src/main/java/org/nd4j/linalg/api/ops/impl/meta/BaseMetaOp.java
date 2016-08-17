package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.MetaOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.impl.grid.BaseGridOp;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseMetaOp extends BaseGridOp {

    protected BaseMetaOp(Op opA, Op opB) {
        super(opA, opB);
    }
}
