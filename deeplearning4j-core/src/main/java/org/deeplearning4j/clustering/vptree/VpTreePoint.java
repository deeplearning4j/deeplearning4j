package org.deeplearning4j.clustering.vptree;

/**
 * An base interface for point in VP Tree.
 * @author Anatoly Borisov
 */
public interface VpTreePoint<T extends VpTreePoint<T>> {
    /**
     * Calculates distance to another point.
     * The metric must hold the following condition for points A, B, C:
     * A.distance(C) <= A.distance(B) + B.distance(C)
     */
    double distance(T p);
}
