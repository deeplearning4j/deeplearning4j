package org.deeplearning4j.models.sequencevectors.graph.exception;

/** Unchecked exception signifying that an error occurred during parsing of text */
public class ParseException extends RuntimeException {
    public ParseException() {
        super();
    }

    public ParseException(String s) {
        super(s);
    }

    public ParseException(String s, Exception e) {
        super(s, e);
    }
}
