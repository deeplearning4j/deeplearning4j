package org.nd4j.autodiff.samediff.internal;

class FrameContext {
    String frameName;
    int iteration;
    FrameIter parentFrame;
    long entryTimestamp;

    public FrameContext(String frameName, int iteration, FrameIter parentFrame) {
        this.frameName = frameName;
        this.iteration = iteration;
        this.parentFrame = parentFrame;
        this.entryTimestamp = System.currentTimeMillis();
    }
}
