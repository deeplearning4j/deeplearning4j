package org.deeplearning4j.spark.api;

/**
 * RepartitionStrategy: different strategies for conducting repartitioning on training data, when repartitioning is required.<br>
 * SparkDefault: repartition using Spark's standard {@code RDD.repartition(int)} method. This results in each value being
 * randomly mapped to a new partition. This results in approximately equal partitions, though random sampling issues can
 * be problematic when the number of elements in a RDD is small<br>
 * Balanced: a custom repartitioning strategy that attempts to ensure that each partition ends up with the correct number
 * of elements. It has a slightly higher overhead (need to count the number of values in each partition) but should be less
 * prone to random sampling variance than the SparkDefault strategy
 *
 *
 * @author Alex Black
 */
public enum RepartitionStrategy {
    SparkDefault, Balanced, ApproximateBalanced

}
