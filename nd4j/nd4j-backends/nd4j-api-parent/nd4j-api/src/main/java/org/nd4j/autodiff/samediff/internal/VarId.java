package org.nd4j.autodiff.samediff.internal;

import lombok.Data;

/**
 * VarId: identifies the value of a variable in a specific frame and frame
 * iteration<br>
 * Note that frames can be nested - which generally represents nested loop
 * situations.<br>
 * Used for 2 places:<br>
 * (a) to identify variables that are available for execution<br>
 * (b) to store results<br>
 */
@Data
public class VarId {
    private String variable;
    private String frame;
    private int iteration;
    private FrameIter parentFrame;

    public VarId(String variable, String frame, int iteration, FrameIter parentFrame) {
        this.variable = variable;
        this.frame = frame;
        this.iteration = iteration;
        this.parentFrame = parentFrame;
    }

    /**
     * Creates the default outer frame
     *
     * @param name the name of the variable ot create an id for
     * @return
     */
    public static VarId createDefault(String name) {
        return new VarId(name, AbstractSession.OUTER_FRAME, 0, null);
    }

    @Override
    public String toString() {
        return "VarId(\"" + variable + "\",\"" + frame + "\"," + iteration + ",parent=" + parentFrame + ")";
    }

    /**
     * @return FrameIter corresponding to the VarId
     */
    public FrameIter toFrameIter() {
        return new FrameIter(frame, iteration, parentFrame);
    }
}
