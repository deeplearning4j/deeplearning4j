package org.nd4j.linalg.profiler.data.array.event;

import org.jetbrains.annotations.NotNull;

public class ComparableStackTrace implements Comparable<ComparableStackTrace> {

    private StackTraceElement[] stackTrace;

    public ComparableStackTrace(StackTraceElement[] stackTrace) {
        this.stackTrace = stackTrace;
    }

    public StackTraceElement[] getStackTrace() {
        return stackTrace;
    }

    @Override
    public int compareTo(@NotNull ComparableStackTrace o) {
        return 0;
    }
}
