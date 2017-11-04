package org.nd4j.autodiff.graph;

import com.google.common.primitives.Ints;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Label;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.graph.api.BaseGraph;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.IGraph;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.graph.exception.NoEdgesException;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static guru.nidi.graphviz.model.Factory.graph;
import static guru.nidi.graphviz.model.Factory.node;
import static guru.nidi.graphviz.model.Link.to;

/** Graph, where all edges and vertices are stored in-memory.<br>
 * Internally, this is a directed graph with adjacency list representation; however, if undirected edges
 * are added, these edges are duplicated internally to allow for fast lookup.<br>
 * Depending on the value of {@code allowMultipleEdges}, this graph implementation may or may not allow
 * multiple edges between any two adjacent nodes. If multiple edges are required (such that two or more distinct edges
 * between vertices X and Y exist simultaneously) then {@code allowMultipleEdges} should be set to {@code true}.<br>
 * As per {@link IGraph}, this graph representation can have arbitrary objects attached<br>
 * Vertices are initialized either directly via list. Edges are added using one of the
 * addEdge methods.
 * @param <V> Type parameter for vertices (opType of objects attached to each vertex)
 * @param <E> Type parameter for edges (opType of objects attached to each edge)
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Slf4j
public class Graph<V, E> extends BaseGraph<V, E> {
    private boolean allowMultipleEdges = true;
    private Map<int[],List<Edge<E>>> edges; //edge[i].get(j).to = k, then edge from i -> k
    private Map<Integer,Vertex<V>> vertices;
    private boolean frozen = false;
    private Map<int[],List<Edge<E>>> incomingEdges;
    private Graph<V,E> graphApply;
    @Getter
    private int nextVertexId;
    private Edge<E> lastEdgeAdded;
    private Edge<E> lastEdgeBeforeLastAdded;
    private Vertex<V> lastVertexAdded;

    public Graph() {
        this(true);
    }

    @SuppressWarnings("unchecked")
    public Graph(boolean allowMultipleEdges) {
        this.allowMultipleEdges = allowMultipleEdges;
        vertices = new TreeMap<>();
        edges = new IntArrayKeyMap<>();
        this.incomingEdges = new IntArrayKeyMap<>();
        nextVertexId = 0;
    }

    public Graph(
            boolean allowMultipleEdges,
            Map<int[], List<Edge<E>>> edges,
            Map<Integer, Vertex<V>> vertices,
            boolean frozen,
            Map<int[], List<Edge<E>>> incomingEdges) {
        this.allowMultipleEdges = allowMultipleEdges;
        this.edges = edges;
        this.vertices = vertices;
        this.frozen = frozen;
        this.incomingEdges = incomingEdges;
        nextVertexId = vertices.size();
    }

    /**
     * Add a vertex to the graph
     * (no effect when frozen)
     * @param vVertex
     */
    public void addVertex(Vertex<V> vVertex) {

        if(frozen) {
            log.trace("Attempted to add vertex to frozen graph " + vVertex);
            return;
        }

        if(vertices.get(vVertex.vertexID()) != null) {
            throw new IllegalArgumentException("Unable to readd vertex. Vertex id must be unique: " + vVertex.getIdx());
        }

        if(vVertex.getIdx() != vertices.size() + 1) {
            throw new IllegalArgumentException("Unable to add vertex. Contiguous id not found");
        }

        //this is for a temporary substitution of
        // the graph
        if(graphApply != null) {
            log.trace("Adding to another graph instead " + vVertex);
            graphApply.addVertex(vVertex);
        }
        else
            this.vertices.put(vVertex.getIdx(),vVertex);

        //track the last vertex added
        lastVertexAdded = vVertex;
    }




    @SuppressWarnings("unchecked")
    public Graph(List<Vertex<V>> vertices, boolean allowMultipleEdges) {

        this.vertices = new TreeMap<>();
        for(Vertex<V> v : vertices)
            this.vertices.put(v.getIdx(),v);
        this.allowMultipleEdges = allowMultipleEdges;
        edges = new HashMap<>();
        this.incomingEdges = new TreeMap<>(Ints.lexicographicalComparator());
        nextVertexId = this.vertices.size();
    }

    public Graph(List<Vertex<V>> vertices) {
        this(vertices, false);
    }


    /**
     * Prevent items from being added to the graph
     */
    public void freeze() {
        frozen = true;
    }

    /**
     * Allow items to be added again
     */
    public void unfreeze() {
        frozen = false;
    }


    @Override
    public int numVertices() {
        return vertices.size();
    }

    @Override
    public Vertex<V> getVertex(int idx) {
        if (idx < 0)
            throw new IllegalArgumentException("Invalid index: " + idx);
        return vertices.get(idx);
    }

    @Override
    public List<Vertex<V>> getVertices(int[] indexes) {
        List<Vertex<V>> out = new ArrayList<>(indexes.length);
        for (int i : indexes)
            out.add(getVertex(i));
        return out;
    }

    @Override
    public List<Vertex<V>> getVertices(int from, int to) {
        if (to < from || from < 0)
            throw new IllegalArgumentException("Invalid range: from=" + from + ", to=" + to);
        List<Vertex<V>> out = new ArrayList<>(to - from + 1);
        for (int i = from; i <= to; i++)
            out.add(getVertex(i));
        return out;
    }

    @Override
    public boolean isFrozen() {
        return frozen;
    }

    @Override
    public void addEdge(Edge<E> edge) {
        if(frozen) {
            log.trace("Attempted to add edge to frozen graph " + edge);
            return;
        }
        else if(graphApply != null) {
            log.trace("Adding edge to apply graph rather than this one " + edge);
            graphApply.addEdge(edge);
            return;
        }


        if(!frozen && Arrays.equals(edge.getFrom(),edge.getTo()))
            throw new IllegalArgumentException("No cycles allowed");

        if(edge.getFrom() == null || edge.getFrom().length < 1)
            throw new ND4JIllegalStateException("From must contain vertices.");

        if(edge.getTo() == null || edge.getTo().length < 1)
            throw new ND4JIllegalStateException("From must contain vertices.");

        for(int i : edge.getFrom())
            if (i <= 0)
                throw new IllegalArgumentException("Invalid edge: " + edge + ", from/to indexes out of range");

        List<Edge<E>> fromList =  edges.get(edge.getFrom());

        if(fromList == null) {
            fromList = new ArrayList<>();
            edges.put(edge.getFrom(),fromList);
        }

        addEdgeHelper(edge, fromList);
        //track last 2 edges added
        if(lastEdgeAdded != null)
            lastEdgeBeforeLastAdded = lastEdgeAdded;
        else
            lastEdgeBeforeLastAdded = edge;

        lastEdgeAdded = edge;

        List<Edge<E>> incomingList =  incomingEdges.get(edge.getTo());
        if(incomingList == null) {
            incomingList = new ArrayList<>();
            incomingEdges.put(edge.getTo(),incomingList);
        }

        addEdgeHelper(edge, incomingList);

        if (edge.isDirected())
            return;


        //Add other way too (to allow easy lookup for undirected edges)
        List<Edge<E>> toList = edges.get(edge.getTo());
        if(toList == null) {
            toList = new ArrayList<>();
            edges.put(edge.getTo(),toList);
        }

        addEdgeHelper(edge, toList);



    }

    @Override
    @SuppressWarnings("unchecked")
    public List<Edge<E>> getEdgesOut(int[] vertex) {
        if (edges.get(vertex) == null)
            return Collections.emptyList();
        return new ArrayList<>(edges.get(vertex));
    }

    @Override
    public int getVertexDegree(int vertex) {
        if (edges.get(vertex) == null)
            return 0;
        return edges.get(vertex).size();
    }

    @Override
    public int getVertexInDegree(int[] vertex) {
        int ret = 0;
        if(!incomingEdges.containsKey(vertex))
            return 0;
        for(Edge<E> edge : incomingEdges.get(vertex)) {
            if(Arrays.equals(edge.getTo(),vertex))
                ret++;
        }

        return ret;
    }



    @Override
    public Vertex<V> getRandomConnectedVertex(int[] vertex, Random rng) throws NoEdgesException {
        if (ArrayUtil.anyLessThan(vertex , 0) || ArrayUtil.anyLargerThan(vertex, vertices.size()))
            throw new IllegalArgumentException("Invalid vertex index: " + vertex);
        if (edges.get(vertex) == null || edges.get(vertex).isEmpty())
            throw new NoEdgesException("Cannot generate random connected vertex: vertex " + vertex
                    + " has no outgoing/undirected edges");
        int connectedVertexNum = rng.nextInt(edges.get(vertex).size());
        Edge<E> edge = edges.get(vertex).get(connectedVertexNum);
        if (Arrays.equals(edge.getFrom(),vertex))
            return vertices.get(edge.getTo()); //directed or undirected, vertex -> x
        else
            return vertices.get(edge.getFrom()); //Undirected edge, x -> vertex
    }

    @Override
    public List<Vertex<V>> getConnectedVertices(int[] vertex) {
        if (ArrayUtil.anyLessThan(vertex, 0) || ArrayUtil.anyLargerThan(vertex,vertices.size()))
            throw new IllegalArgumentException("Invalid vertex index: " + vertex);

        if (edges.get(vertex) == null)
            return Collections.emptyList();
        List<Vertex<V>> list = new ArrayList<>(edges.get(vertex).size());
        for (Edge<E> edge : edges.get(vertex)) {
            list.add(vertices.get(edge.getTo()));
        }
        return list;
    }

    @Override
    public int[] getConnectedVertexIndices(int[] vertex) {
        if(!edges.containsKey( edges.get(vertex)))
            return new int[0];

        List<Integer> ret = new ArrayList<>();
        List<Edge<E>> edgeList = edges.get(vertex);
        for (int i = 0; i < edgeList.size(); i++) {
            Edge<E> e = edges.get(vertex).get(i);
            int[] arr = (Arrays.equals(e.getFrom(),vertex) ? e.getTo() : e.getFrom());
            ret.addAll(Ints.asList(arr));
        }

        return Ints.toArray(ret);
    }

    private void addEdgeHelper(Edge<E> edge, List<Edge<E>> list) {
        if (!allowMultipleEdges) {
            //Check to avoid multiple edges
            boolean duplicate = false;

            if (edge.isDirected()) {
                for (Edge<E> e : list) {
                    if (Arrays.equals(e.getTo(),edge.getTo())) {
                        duplicate = true;
                        break;
                    }
                }
            } else {
                for (Edge<E> e : list) {
                    if ((Arrays.equals(e.getFrom(),edge.getFrom()) && Arrays.equals(e.getTo(), edge.getTo()))
                            || (e.getTo() == edge.getFrom() && e.getFrom() == edge.getTo())) {
                        duplicate = true;
                        break;
                    }
                }
            }

            if (!duplicate) {
                list.add(edge);
            }

        } else {
            //allow multiple/duplicate edges
            list.add(edge);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Graph {");
        sb.append("\nVertices {");
        for (Vertex<V> v : vertices.values()) {
            sb.append("\n\t").append(v);
        }
        sb.append("\n}");
        sb.append("\nEdges {");
        for (int[] i : edges.keySet()) {
            sb.append("\n\t");
            sb.append(Arrays.toString(i)).append(":");
            for (Edge<E> e : edges.get(i)) {
                sb.append(" ").append(e).append("\n");

            }
        }



        sb.append("\n}");


        sb.append("\n Incoming edges {");
        for (int[] i : incomingEdges.keySet()) {
            sb.append("\n\t");
            sb.append(Arrays.toString(i)).append(":");
            for (Edge<E> e : incomingEdges.get(i)) {
                sb.append(" ").append(e).append("\n");

            }
        }

        sb.append("\n}");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Graph<?, ?> graph = (Graph<?, ?>) o;
        return toString().equals(graph.toString());
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (edges != null ? edges.hashCode() : 0);
        result = 31 * result + (vertices != null ? vertices.hashCode() : 0);
        return result;
    }

    /**
     * Save the graph to a file with graphviz
     * @param path
     * @throws IOException
     */
    public void print(File path) throws IOException {
        guru.nidi.graphviz.model.Graph g = graph("example5").directed();
        for(List<Edge<E>> edgeList : getEdges().values())
            for(Edge<E> edge : edgeList) {
                NDArrayVertex vertex = (NDArrayVertex) getVertex(edge.getFrom()[0]);
                NDArrayVertex v2 = (NDArrayVertex) getVertex(edge.getTo()[0]);
                OpState opState = (OpState) edge.getValue();
                g = g.with(node(String.valueOf(vertex.vertexID()))
                        .with(Label.of(String.valueOf(vertex.getValue().getVarName())))
                        .link(to(node(String.valueOf(v2.vertexID()))
                                .with(Label.of(v2.getValue().getVarName())))
                                .with(Label.of(opState.getOpName()))));


            }

        for(Vertex<V> vertex : getVertices().values()) {
            if(!edges.containsKey(vertex.getIdx())) {
                SDVariable vertex1 = (SDVariable) vertex.getValue();
                g = g.with(node(String.valueOf(vertex.vertexID())).with(Label.of(vertex1.getVarName())));
            }
        }

        Graphviz viz = Graphviz.fromGraph(g);
        viz.width(15000).height(800).scale(0.5)
                .render(Format.PNG).toFile(path);

    }

    public int nextVertexId() {
        if(frozen) {
            return nextVertexId;
        }
        else if(nextVertexId > vertices.size())
            throw new ND4JIllegalStateException("Unable to increment vertex id.  No vertices added.");
        return ++nextVertexId; // Make vertexIds start at 1 to uncover array indexing issues.
    }
}




