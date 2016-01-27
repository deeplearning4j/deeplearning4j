package org.deeplearning4j.nn.conf.inputs;


/** InvalidInputTypeException: Thrown if the GraphVertex cannot handle the type of input provided */
public class InvalidInputTypeException extends Exception {

    public InvalidInputTypeException(String message) {
        super(message);
    }

    public InvalidInputTypeException(String message, Throwable cause) {
        super(message, cause);
    }

    public InvalidInputTypeException(Throwable cause) {
        super(cause);
    }
}
