package org.nd4j.autodiff.samediff.internal;

import java.util.ArrayList;
import java.util.List;

public class SwitchResult {
    String operationName;
    String branchTaken;
    Object predicateValue;
    int outputIndex = -1;
    String frameContext;
    List<String> affectedVariables = new ArrayList<>();

    @Override
    public String toString() {
        return String.format("Switch[%s]: %s branch (predicate: %s, outputs: %s)",
                operationName, branchTaken, predicateValue, affectedVariables.size());
    }
}
