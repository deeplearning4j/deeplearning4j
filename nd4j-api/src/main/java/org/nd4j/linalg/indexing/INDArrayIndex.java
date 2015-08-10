package org.nd4j.linalg.indexing;

/**
 * @author Adam Gibson
 */
public interface INDArrayIndex {
    int end();

    int offset();

    int length();

    int[] indices();

    void reverse();

    boolean isInterval();

    void setInterval(boolean isInterval);
}
