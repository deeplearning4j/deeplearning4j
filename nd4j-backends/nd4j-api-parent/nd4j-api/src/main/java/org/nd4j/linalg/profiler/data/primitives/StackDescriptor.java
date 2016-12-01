package org.nd4j.linalg.profiler.data.primitives;

import lombok.Getter;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

/**
 * @author raver119
 */
public class StackDescriptor {
    @Getter protected StackTraceElement stackTrace[];

    public StackDescriptor(@NonNull StackTraceElement stack[]) {
        this.stackTrace = stack;
        ArrayUtils.reverse(this.stackTrace);
    }

    public String getEntryName() {
        return getElementName(0);
    }

    public String getElementName(int idx) {
        return stackTrace[idx].getClassName() + "." + stackTrace[idx].getMethodName() + ":" + stackTrace[idx].getLineNumber();
    }

    public int size() {
        return stackTrace.length;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        StackDescriptor that = (StackDescriptor) o;

        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        return Arrays.equals(stackTrace, that.stackTrace);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(stackTrace);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Stack trace: \n");

        for (int i = 0; i < size(); i++) {
            builder.append("         ").append(i).append(": ").append(getElementName(i)).append("\n");
        }

        return builder.toString();
    }
}
