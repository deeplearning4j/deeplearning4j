package org.nd4j.autodiff.samediff;

/**
 * Information about frame context and transitions
 */
class FrameInfo {
    String inputFrame;
    int inputIteration;
    String inputParentFrame;
    String outputFrame;
    int outputIteration;
    String outputParentFrame;
    FrameTransition frameTransition = FrameTransition.NONE;
    String targetFrame; // For Enter operations
}
