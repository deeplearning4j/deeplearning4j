package org.nd4j.autodiff.samediff.internal;

public class ExecStepDependencyTracker extends DependencyTracker<ExecStep,ExecStep> {

    @Override
    protected ExecStep createDependeeForFrame(ExecStep originalDependee, String varName, FrameIter targetFrame) {
        if (originalDependee == null || targetFrame == null) {
            return null;
        }

        // Create a new ExecStep for the target frame
        return new ExecStep(originalDependee.getType(), varName, targetFrame);
    }
}
