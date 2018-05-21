package org.nd4j.linalg.exception;

/**
 * This exception is thrown once if INDArray length exceeds Integer.MAX_VALUE
 *
 * @author raver119@gmail.com
 */
public class ND4JArraySizeException extends ND4JException {
    public ND4JArraySizeException() {
        super("INDArray length is too big to fit into JVM array");
    }
}
