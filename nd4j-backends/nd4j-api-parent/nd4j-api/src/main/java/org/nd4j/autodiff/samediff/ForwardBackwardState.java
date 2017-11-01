package org.nd4j.autodiff.samediff;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.linalg.api.ops.Op;

import java.io.Serializable;
import java.util.List;

/**
 *
 */
@Builder
@EqualsAndHashCode(callSuper = false)
@Getter
@Setter
public class ForwardBackwardState implements Serializable {

    private List<DifferentialFunction> forward;
    private List<DifferentialFunction> backward;
    private List<SDVariable> forwardVariable;
    private List<SDVariable> backwardVariable;
    private Op forwardOp;
    private Op backwardOp;
    private OpExecAction forwardOpExecAction;


}
