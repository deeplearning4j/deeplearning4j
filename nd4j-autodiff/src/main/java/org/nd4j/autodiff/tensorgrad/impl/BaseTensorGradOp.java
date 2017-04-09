package org.nd4j.autodiff.tensorgrad.impl;

import lombok.AllArgsConstructor;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.tensorgrad.TensorGradOp;

/**
 * Created by agibsonccc on 4/9/17.
 */
@AllArgsConstructor
public abstract class BaseTensorGradOp implements TensorGradOp {
    protected OpState opState;


}
