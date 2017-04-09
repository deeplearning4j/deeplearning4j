package org.nd4j.autodiff.tensorgrad;

import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.ops.Op;

/**
 * Created by agibsonccc on 4/9/17.
 */
public interface TensorGradOp {

    OpState opState();

    String name();

    TensorGradVariable compute(TensorGradVariable...variables);

    Op createOp();
}
