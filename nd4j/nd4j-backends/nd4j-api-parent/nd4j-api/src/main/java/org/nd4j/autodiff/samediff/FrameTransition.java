package org.nd4j.autodiff.samediff;

/**
 * Types of frame transitions
 */
enum FrameTransition {
    NONE,           // No frame change
    ENTER,          // Enter a new frame (loop/conditional)
    EXIT,           // Exit current frame
    NEXT_ITERATION, // Move to next iteration in current frame
    SWITCH,         // Conditional branch
    MERGE,          // Merge conditional branches
    LOOP_CONDITION  // Loop condition check
}
