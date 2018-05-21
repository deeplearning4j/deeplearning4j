package org.nd4j.linalg.exception;

/**
 * This is temporary exception, and is used to highlight missing ComplexNumber implementation
 *
 * @author raver119@gmail.com
 */
public class ND4JComplexNumbersNotSupportedException extends ND4JException {
    public ND4JComplexNumbersNotSupportedException() {
        super("Complex numbers support missing yet");
    }
}
