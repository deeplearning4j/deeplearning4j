package org.deeplearning4j.clustering.algorithm;

public enum Distance {
    EUCLIDIAN("euclidean"),
    COSINE_DISTANCE("cosinedistance"),
    COSINE_SIMILARITY("cosinesimilarity"),
    MANHATTAN("manhattan"),
    DOT("dot"),
    JACCARD("jaccard"),
    HAMMING("hamming");

    private String functionName;
    private Distance(String name) {
        functionName = name;
    }

    @Override
    public String toString() {
        return functionName;
    }
}
