package org.deeplearning4j.text.labels;

/**
 * @author raver119@gmail.com
 */
public interface LabelsProvider {

    String nextLabel();

    String getLabel(int index);
}
