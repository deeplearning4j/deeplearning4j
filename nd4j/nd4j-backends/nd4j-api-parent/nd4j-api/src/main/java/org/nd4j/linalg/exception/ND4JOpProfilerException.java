package org.nd4j.linalg.exception;

/**
 * ND4JOpProfilerException: Thrown by the op profiler (if enabled) for example on NaN panic
 *
 * @author Alex Black
 */
public class ND4JOpProfilerException extends ND4JIllegalStateException {
    public ND4JOpProfilerException() {
    }
    public ND4JOpProfilerException(String message) {
        super(message);
    }

    public ND4JOpProfilerException(String message, Throwable cause) {
        super(message, cause);
    }

    public ND4JOpProfilerException(Throwable cause) {
        super(cause);
    }
}
