package org.deeplearning4j.models.sequencevectors.graph.enums;

/**
 * This enum is used in PopularityWalker, and it defines which nodes will be considered for next hop.
 * MAXIMUM: top-popularity nodes will be considered.
 * AVERAGE: nodes in the middle of possible selections will be considered.
 * MINIMUM: low-popularity nodes will be considered.
 *
 * @author raver119@gmail.com
 */
public enum PopularityMode {
    MAXIMUM,
    AVERAGE,
    MINIMUM,
}
