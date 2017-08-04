package org.deeplearning4j.datasets.datavec.exception;

/**
 * Unchecked exception, thrown to signify that a zero-length sequence data set was encountered.
 */
public class ZeroLengthSequenceException extends RuntimeException {
    public ZeroLengthSequenceException() {
        this("");
    }

    public ZeroLengthSequenceException(String type) {
        super(String.format("Encountered zero-length %ssequence", type.equals("") ? "" : type + " "));
    }
}
