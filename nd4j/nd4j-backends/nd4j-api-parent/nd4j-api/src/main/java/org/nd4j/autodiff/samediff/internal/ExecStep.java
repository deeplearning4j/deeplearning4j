package org.nd4j.autodiff.samediff.internal;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;

/**
 * ExecStep represents a single execution step, for a single op (or
 * variable/constant etc) at a specific frame/iteration
 */
@Getter
@EqualsAndHashCode
class ExecStep {
    protected final ExecType type;
    protected final String name;
    protected final FrameIter frameIter;

    protected ExecStep(@NonNull ExecType execType, @NonNull String name, FrameIter frameIter) {
        this.type = execType;
        this.name = name;
        this.frameIter = frameIter;
    }

    protected VarId toVarId() {
        return new VarId(name, frameIter.getFrame(), frameIter.getIteration(), frameIter.getParentFrame());
    }

    @Override
    public String toString() {
        return "ExecStep(" + type + ",name=\"" + name + "\"," + frameIter + ")";
    }

}
