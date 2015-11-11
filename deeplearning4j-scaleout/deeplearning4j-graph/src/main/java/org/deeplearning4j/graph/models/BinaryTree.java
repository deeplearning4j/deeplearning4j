package org.deeplearning4j.graph.models;

public interface BinaryTree {

    long getCode(int element);

    int getCodeLength(int element);

    String getCodeString(int element);

    int[] getPathInnerNodes(int element);
}
