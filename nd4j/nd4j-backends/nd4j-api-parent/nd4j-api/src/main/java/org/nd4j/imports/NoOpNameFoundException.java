package org.nd4j.imports;

import org.nd4j.linalg.exception.ND4JIllegalStateException;

public class NoOpNameFoundException extends ND4JIllegalStateException {
    public NoOpNameFoundException() {
    }

    public NoOpNameFoundException(String message) {
        super(message);
    }
}
