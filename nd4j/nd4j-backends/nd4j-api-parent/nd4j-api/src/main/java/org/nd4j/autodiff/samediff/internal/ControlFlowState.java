package org.nd4j.autodiff.samediff.internal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Enhanced helper classes and methods for control flow tracking
 */

public class ControlFlowState {
    int executionCount = 0;
    String currentFrame;
    int currentIteration;
    List<String> frameTransitions = new ArrayList<>();
    List<SwitchResult> switchDecisions = new ArrayList<>();
    List<MergeResult> mergeDecisions = new ArrayList<>();
    Map<String, Object> additionalMetadata = new HashMap<>();
}
