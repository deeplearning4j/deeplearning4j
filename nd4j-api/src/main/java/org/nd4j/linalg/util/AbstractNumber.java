package org.nd4j.linalg.util;

/**
 *
 * @author bogachenko
 */

public interface AbstractNumber {
    AbstractNumber add(AbstractNumber b);
    AbstractNumber sub(AbstractNumber b);
    AbstractNumber mult(AbstractNumber b);
    AbstractNumber div(AbstractNumber b);
}