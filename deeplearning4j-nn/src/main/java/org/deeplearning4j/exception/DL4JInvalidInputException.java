package org.deeplearning4j.exception;

/**
 * DL4J Exception thrown when invalid input is provided (wrong rank, wrong size, etc)
 *
 * @author Alex Black
 */
public class DL4JInvalidInputException extends DL4JException {

    public DL4JInvalidInputException() {}

    public DL4JInvalidInputException(String message) {
        super(message);
    }

    public DL4JInvalidInputException(String message, Throwable cause) {
        super(message, cause);
    }

    public DL4JInvalidInputException(Throwable cause) {
        super(cause);
    }
}
