package org.deeplearning4j.util;

public class ModelGuesserException extends Exception {

    public ModelGuesserException(String message) {
        super(message);
    }

    public ModelGuesserException(String message, Throwable cause) {
        super(message, cause);
    }

    public ModelGuesserException(Throwable cause) {
        super(cause);
    }
}
