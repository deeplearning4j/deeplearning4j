package org.deeplearning4j.graph.api;

public abstract class BaseGraph<V,E> implements Graph<V,E> {


    public void addEdge(int from, int to, E value, boolean directed){
        addEdge(new Edge<>(from,to,value,directed));
    }

}
