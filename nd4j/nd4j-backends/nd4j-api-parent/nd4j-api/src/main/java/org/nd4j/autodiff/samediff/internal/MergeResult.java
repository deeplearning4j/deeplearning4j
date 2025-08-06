package org.nd4j.autodiff.samediff.internal;

public class MergeResult {
    String operationName;
    int selectedInputIndex = -1;
    int totalInputs;
    String selectedInputSource;
    Object mergedValue;
    String frameContext;

    @Override
    public String toString() {
        return String.format("Merge[%s]: Selected input %d of %d (source: %s)",
                operationName, selectedInputIndex, totalInputs, selectedInputSource);
    }
}
