package org.deeplearning4j.graph.api;

/**
 * Created by Alex on 9/11/2015.
 */
public abstract class BaseGraph<V,E> implements Graph<V,E> {


    public void addEdge(int from, int to, E value, boolean directed){
        addEdge(new Edge<>(from,to,value,directed));
    }

}
