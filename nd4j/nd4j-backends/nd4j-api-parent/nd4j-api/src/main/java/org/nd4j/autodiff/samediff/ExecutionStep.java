package org.nd4j.autodiff.samediff;

import org.nd4j.autodiff.samediff.internal.ExecType;
import org.nd4j.autodiff.samediff.internal.FrameIter;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a single execution step in the visualization
 */
public class ExecutionStep {
    private final int stepNumber;
    private final String timestamp;
    private final ExecType type;
    private final String name;
    private final String frame;
    private final int iteration;
    private final FrameIter parentFrame;
    private final List<String> inputs;
    private final List<String> outputs;
    private final String status;

    public ExecutionStep(int stepNumber, String timestamp,
                         ExecType type, String name,
                         String frame, int iteration,
                         FrameIter parentFrame,
                         List<String> inputs, List<String> outputs, String status) {
        this.stepNumber = stepNumber;
        this.timestamp = timestamp;
        this.type = type;
        this.name = name;
        this.frame = frame;
        this.iteration = iteration;
        this.parentFrame = parentFrame;
        this.inputs = inputs != null ? new ArrayList<>(inputs) : new ArrayList<>();
        this.outputs = outputs != null ? new ArrayList<>(outputs) : new ArrayList<>();
        this.status = status;
    }

    // Getters
    public int getStepNumber() {
        return stepNumber;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public ExecType getType() {
        return type;
    }

    public String getName() {
        return name;
    }

    public String getFrame() {
        return frame;
    }

    public int getIteration() {
        return iteration;
    }

    public FrameIter getParentFrame() {
        return parentFrame;
    }

    public List<String> getInputs() {
        return inputs;
    }

    public List<String> getOutputs() {
        return outputs;
    }

    public String getStatus() {
        return status;
    }
}
