package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;

@AllArgsConstructor
@Data
@EqualsAndHashCode(callSuper = true)
@SuperBuilder
public class OpDep extends Dep {
    protected String opName;
    protected int iter;

    protected OpDep(@NonNull String opName, @NonNull String frame, int iter, FrameIter parentFrame) {
        this.opName = opName;
        this.frame = frame;
        this.iter = iter;
        this.parentFrame = parentFrame;
    }

    @Override
    public String toString() {
        return "OpDep(" + opName + ",frame=" + frame + ",iter=" + iter + (parentFrame == null ? "" : ",parent=" + parentFrame) + ")";
    }
}
