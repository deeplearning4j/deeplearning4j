package org.deeplearning4j.graph.api;

/**When walking a graph, how should we handle disconnected nodes?
 * i.e., those without any outgoing (directed) or undirected edges
 */
public enum NoEdgeHandling {
    SELF_LOOP_ON_DISCONNECTED, EXCEPTION_ON_DISCONNECTED

}
