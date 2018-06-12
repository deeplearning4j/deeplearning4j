package org.nd4j.linalg.exception;

public class ND4UnresolvedOutputVariables extends ND4JException {
    public ND4UnresolvedOutputVariables(String message) {
        super(message);
    }

    public ND4UnresolvedOutputVariables(String message, Throwable cause) {
        super(message, cause);
    }
}
