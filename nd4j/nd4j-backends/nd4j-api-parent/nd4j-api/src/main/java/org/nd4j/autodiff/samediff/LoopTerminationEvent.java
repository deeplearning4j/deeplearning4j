package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class LoopTerminationEvent {
    private String frameName;
    private int iteration;
    private long timestamp;
    private TerminationType terminationType;
    private String triggerOperation;
    private Object terminationValue;
    private String terminationReason;
    private Map<String, Object> contextData = new HashMap<>();
    private List<String> affectedVariables = new ArrayList<>();
    private LoopState loopStateAtTermination;
    private boolean wasEarlyTermination;
    private String earlyTerminationCause;
}
