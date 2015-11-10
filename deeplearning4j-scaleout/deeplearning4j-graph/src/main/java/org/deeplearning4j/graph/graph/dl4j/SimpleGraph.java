package org.deeplearning4j.graph.graph.dl4j;

import org.deeplearning4j.graph.api.*;
import org.deeplearning4j.graph.exception.NoEdgesException;
import org.deeplearning4j.graph.vertexfactory.VertexFactory;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class SimpleGraph<V, E> extends BaseGraph<V,E> {
    private boolean allowMultipleEdges;

    private List<Edge<E>>[] edges;  //edgeList.get(i).get(j).to = k, then edge from i -> k

    private List<Vertex<V>> vertices;

    @SuppressWarnings("unchecked")
    public SimpleGraph(int numVertices, boolean allowMultipleEdges, VertexFactory<V> vertexFactory){
        if(numVertices <= 0 ) throw new IllegalArgumentException();
        this.allowMultipleEdges = allowMultipleEdges;

        vertices = new ArrayList<>(numVertices);
        for( int i=0; i<numVertices; i++ ) vertices.add(vertexFactory.create(i));

        edges = (List<Edge<E>>[]) Array.newInstance(List.class,numVertices);
    }

    @SuppressWarnings("unchecked")
    public SimpleGraph(List<Vertex<V>> vertices, boolean allowMultipleEdges ){
        this.vertices = new ArrayList<>(vertices);
        this.allowMultipleEdges = allowMultipleEdges;
        edges = (List<Edge<E>>[]) Array.newInstance(List.class,vertices.size());
    }

    @Override
    public int numVertices() {
        return vertices.size();
    }

    @Override
    public Vertex<V> getVertex(int idx) {
        if(idx < 0 || idx >= vertices.size() ) throw new IllegalArgumentException("Invalid index: " + idx);
        return vertices.get(idx);
    }

    @Override
    public List<Vertex<V>> getVertices(int[] indexes) {
        List<Vertex<V>> out = new ArrayList<>(indexes.length);
        for(int i : indexes) out.add(getVertex(i));
        return out;
    }

    @Override
    public List<Vertex<V>> getVertices(int from, int to) {
        if(to < from || from < 0 || to >= vertices.size())
            throw new IllegalArgumentException("Invalid range: from="+from + ", to="+to);
        List<Vertex<V>> out = new ArrayList<>(to-from+1);
        for(int i=from; i<=to; i++ ) out.add(getVertex(i));
        return out;
    }

    @Override
    public void addEdge(Edge<E> edge) {
        if(edge.getFrom() < 0 || edge.getTo() >= vertices.size() )
            throw new IllegalArgumentException("Invalid edge: " + edge + ", from/to indexes out of range");

        List<Edge<E>> fromList = edges[edge.getFrom()];
        if(fromList == null){
            fromList = new ArrayList<>();
            edges[edge.getFrom()] = fromList;
        }
        addEdgeHelper(edge,fromList);

        if(edge.isDirected()) return;

        //Add other way too (to allow easy lookup for undirected edges)
        List<Edge<E>> toList = edges[edge.getTo()];
        if(toList == null){
            toList = new ArrayList<>();
            edges[edge.getTo()] = toList;
        }
        addEdgeHelper(edge,toList);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<Edge<E>> getEdgesOut(int vertex) {
        if(edges[vertex] == null ) return Collections.emptyList();
        return new ArrayList<>(edges[vertex]);
    }

    @Override
    public int getNumEdgesOut(int vertex){
        if(edges[vertex] == null) return 0;
        return edges[vertex].size();
    }

    @Override
    public Vertex<V> getRandomConnectedVertex(int vertex, Random rng) throws NoEdgesException {
        if(vertex < 0 || vertex >= vertices.size() ) throw new IllegalArgumentException("Invalid vertex index: " + vertex);
        if(edges[vertex] == null || edges[vertex].size() == 0)
            throw new NoEdgesException("Cannot generate random connected vertex: vertex " + vertex + " has no outgoing/undirected edges");
        int connectedVertexNum = rng.nextInt(edges[vertex].size());
        Edge<E> edge = edges[vertex].get(connectedVertexNum);
        if(edge.getFrom() == vertex ) return vertices.get(edge.getTo());    //directed or undirected, vertex -> x
        else return vertices.get(edge.getFrom());   //Undirected edge, x -> vertex
    }

    @Override
    public List<Vertex<V>> getConnectedVertices(int vertex) {
        if(vertex < 0 || vertex >= vertices.size()) throw new IllegalArgumentException("Invalid vertex index: " + vertex);

        if(edges[vertex] == null) return Collections.emptyList();
        List<Vertex<V>> list = new ArrayList<>(edges[vertex].size());
        for(Edge<E> edge : edges[vertex]){
            list.add(vertices.get(edge.getTo()));
        }
        return list;
    }

    @Override
    public int[] getConnectedVertexIndices(int vertex){
        int[] out = new int[(edges[vertex] == null ? 0 : edges[vertex].size())];
        if(out.length == 0 ) return out;
        for(int i=0; i<out.length; i++ ){
            Edge<E> e = edges[vertex].get(i);
            out[i] = (e.getFrom() == vertex ? e.getTo() : e.getFrom() );
        }
        return out;
    }

    private void addEdgeHelper(Edge<E> edge, List<Edge<E>> list ){
        if(!allowMultipleEdges){
            //Check to avoid multiple edges
            boolean duplicate = false;

            if(edge.isDirected()){
                for(Edge<E> e : list ){
                    if(e.getTo() == edge.getTo()){
                        duplicate = true;
                        break;
                    }
                }
            } else {
                for(Edge<E> e : list ){
                    if((e.getFrom() == edge.getFrom() && e.getTo() == edge.getTo())
                            || (e.getTo() == edge.getFrom() && e.getFrom() == edge.getTo())){
                        duplicate = true;
                        break;
                    }
                }
            }

            if(!duplicate){
                list.add(edge);
            }
        } else {
            //allow multiple/duplicate edges
            list.add(edge);
        }
    }


    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("SimpleGraph {");
        sb.append("\nVertices {");
        for(Vertex<V> v : vertices){
            sb.append("\n\t").append(v);
        }
        sb.append("\n}");
        sb.append("\nEdges {");
        for( int i=0; i<edges.length; i++ ){
            sb.append("\n\t");
            if(edges[i] == null) continue;
            sb.append(i).append(":");
            for(Edge<E> e : edges[i]){
                sb.append(" ").append(e);
            }
        }
        sb.append("\n}");
        sb.append("\n}");
        return sb.toString();
    }
}
