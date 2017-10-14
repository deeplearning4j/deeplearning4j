package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.*;

import java.util.*;

/**
 * Graph data structure for tensors
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Data
public class SDGraph extends Graph<NDArrayInformation,OpState> {

    protected SameDiff sameDiff;

    public SDGraph(boolean allowMultipleEdges) {
        super(allowMultipleEdges);
    }

    public SDGraph(SDGraph gradGraph) {
        setEdges(gradGraph.getEdges());
        setVertices(gradGraph.getVertices());
        setFrozen(gradGraph.isFrozen());
        setIncomingEdges(gradGraph.getIncomingEdges());
        setGraphApply(gradGraph.getGraphApply());
    }


    @Builder
    private SDGraph(boolean allowMultipleEdges,
                    Map<int[], List<Edge<OpState>>> edges,
                    Map<Integer, Vertex<NDArrayInformation>> vertices,
                    boolean frozen,
                    Map<int[], List<Edge<OpState>>> incomingEdges,
                    SameDiff sameDiff) {
        super(allowMultipleEdges, edges, vertices, frozen, incomingEdges);
        this.sameDiff = sameDiff;
    }

    /**
     * Add a vertex to the graph
     * (no effect when frozen)
     *
     * @param ndArrayInformationVertex
     */
    @Override
    public void addVertex(Vertex<NDArrayInformation> ndArrayInformationVertex) {
        if(getGraphApply() != null) {
            ndArrayInformationVertex.setIdx(getGraphApply().getNextVertexId());
        }

        super.addVertex(ndArrayInformationVertex);
    }

    @Override
    public SDGraph getGraphApply() {
        return (SDGraph) super.getGraphApply();
    }

    @Override
    public String toString() {
        return super.toString();
    }

    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }

    @Override
    public int hashCode() {
        return super.hashCode();
    }

    /**
     * Get the output vertices
     * @return
     */
    public List<NDArrayInformation> getOutputs() {
        List<NDArrayInformation> ret = new ArrayList<>();
        for (int i : getVertices().keySet()) {
            if (getEdgesOut(new int[]{i}).size() < 1)
                ret.add(getVertex(i).getValue());
        }

        return ret;
    }

    /**
     * Get the input vertices
     * @return
     */
    public List<NDArrayInformation> getInputs() {
        List<NDArrayInformation> ret = new ArrayList<>();
        for (int i : getVertices().keySet()) {
            int[] key = {i};
            if (getVertexInDegree(key) < 1) {
                ret.add(getVertex(i).getValue());
            }
        }

        return ret;
    }



    /**
     *
     * @return
     */
    public OpExecOrder getOpOrder(boolean reverse) {
        int[] order = topologicalSort(reverse);
        Set<OpState> seenStates = new HashSet<>();
        if(reverse) {
            List<OpExecAction> forwardActions = getOpOrder().getActions();
            Map<Integer,OpExecAction> opExecActionMap = new HashMap<>();
            //size the vertex id relative to where it would be encountered
            //in a reverse order traversal
            final Map<Integer,Integer> normalForwardOrderMap = new HashMap<>();
            for(int i = forwardActions.size() - 1,j = 0; i >= 0; i--,j++) {
                OpExecAction currAction = forwardActions.get(i);
                normalForwardOrderMap.put(currAction.getOutputId(),j);
                opExecActionMap.put(currAction.getOutputId(),currAction);
            }


            PriorityQueue<NDArrayVertex> depthQueue = new PriorityQueue<>(numVertices(), new Comparator<NDArrayVertex>() {
                @Override
                public int compare(NDArrayVertex o1, NDArrayVertex o2) {
                    if(o1.depth() == o2.depth()) {
                        Integer o1Compare = normalForwardOrderMap.get(o1.vertexID());
                        Integer o2Compare = normalForwardOrderMap.get(o2.vertexID());
                        if(o1Compare != null && o2Compare != null) {
                            return Ints.compare(o1Compare, o2Compare);
                        }
                        else {
                            System.out.println(o1.vertexID() + " was null or " + o2.vertexID() + " was null during comparison");
                        }
                    }
                    return Ints.compare(-o1.depth(),-o2.depth());
                }
            });

            for(int i : order) {
                NDArrayVertex vertex = (NDArrayVertex) getVertex(i);
                depthQueue.add(vertex);
            }

            List<OpExecAction> ret = new ArrayList<>();
            while(!depthQueue.isEmpty()) {
                NDArrayVertex ndArrayVertex = depthQueue.poll();
                OpExecAction action = opExecActionMap.get(ndArrayVertex.vertexID());
                //no op means it was a variable
                if(action != null && !seenStates.contains(action.getOpState())) {
                    ret.add(action);
                    seenStates.add(action.getOpState());
                }

            }


            return OpExecOrder.builder().actions(ret).build();

        }
        else {
            List<OpExecAction> ret = new ArrayList<>();

            //iterate over op execution order skipping
            // nodes that are only inputs
            //the goal is to get all of the needed op executions
            for (int i = 0; i < order.length; i++) {
                //skip vertices that are only inputs
                if (getVertexInDegree(new int[]{order[i]}) < 1) {
                    continue;
                }

                int numInputs = Math.max(1, getVertexInDegree(new int[]{order[i]}));
                int inputsCount = 0;
                List<Integer> inputIdsList = new ArrayList<>();
                List<Edge<OpState>> inputOpStates = getIncomingEdges().get(new int[]{order[i]});
                List<NDArrayInformation> inputInfo = new ArrayList<>();
                //get the inputs for this this output array
                for (Edge<OpState> edge : inputOpStates) {
                    inputIdsList.addAll(Ints.asList(edge.getFrom()));
                    for(int input : edge.getFrom())  {
                        Preconditions.checkNotNull(getInformationFor(input));
                        inputInfo.add(getInformationFor(input));
                        inputsCount++;
                    }
                }

                Preconditions.checkState(inputsCount == numInputs, "Not all inputs were filled.");
                //add edges
                Edge<OpState> opStateEdge = inputOpStates.get(0);
                if(!seenStates.contains(opStateEdge.getValue())) {
                    ret.add(OpExecAction.builder()
                            .output(opStateEdge.getValue().getResults()[0])
                            .opState(opStateEdge.getValue())
                            .inputs(inputInfo.toArray(new NDArrayInformation[inputInfo.size()]))
                            .inputsIds(Ints.toArray(inputIdsList))
                            .outputId(order[i])
                            .build());
                    seenStates.add(opStateEdge.getValue());
                }
            }


            Collections.sort(ret, new Comparator<OpExecAction>() {
                @Override
                public int compare(OpExecAction o1, OpExecAction o2) {
                    return Integer.compare(o1.getOutputId(),o2.getOutputId());
                }
            });

            return OpExecOrder.builder().actions(ret).build();
        }

    }


    /**
     *
     * @return
     */
    public OpExecOrder getOpOrder() {
        return getOpOrder(false);
    }

    /**
     * {@link NDArrayInformation}
     * accessor for a given vertex
     * @param vertex the vertex id
     * @return the information for the vertex
     */
    public NDArrayInformation getInformationFor(int vertex) {
        Vertex<NDArrayInformation> ndArrayInformation = getVertex(vertex);
        if(ndArrayInformation == null)
            return null;
        return ndArrayInformation.getValue();
    }


    /**
     * Topological sort over vertex ids
     * @return
     */
    public int[] topologicalSort(boolean reverse) {
        List<Integer> vertices = new ArrayList<>(getVertices().keySet());
        Collections.sort(vertices);
        List<Integer> retList = new ArrayList<>();

        if(reverse) {
            List<OpExecAction> forwardActions = getOpOrder().getActions();
            //size the vertex id relative to where it would be encountered
            //in a reverse order traversal
            final Map<Integer,Integer> normalForwardOrderMap = new HashMap<>();
            for(int i = forwardActions.size() - 1,j = 0; i >= 0; i--,j++) {
                OpExecAction currAction = forwardActions.get(i);
                normalForwardOrderMap.put(currAction.getOutputId(),j);
            }

            Collections.reverse(vertices);

            PriorityQueue<NDArrayVertex> depthQueue = new PriorityQueue<>(numVertices(), new Comparator<NDArrayVertex>() {
                @Override
                public int compare(NDArrayVertex o1, NDArrayVertex o2) {
                    if(o1.depth() == o2.depth()) {
                        Integer o1Compare = normalForwardOrderMap.get(o1.vertexID());
                        Integer o2Compare = normalForwardOrderMap.get(o2.vertexID());
                        if(o1Compare != null && o2Compare != null) {
                            return Ints.compare(o1Compare, o2Compare);
                        }
                        else {
                            System.out.println(o1.vertexID() + " was null or " + o2.vertexID() + " was null during comparison");
                        }
                    }
                    return Ints.compare(-o1.depth(),-o2.depth());
                }
            });

            for(int i : vertices) {
                NDArrayVertex vertex = (NDArrayVertex) getVertex(i);
                depthQueue.add(vertex);
            }

            while(!depthQueue.isEmpty()) {
                NDArrayVertex vertex =  depthQueue.poll();
                retList.add(vertex.vertexID());
            }


        }
        else {
            LinkedList<int[]> noIncoming = new LinkedList<>();
            Map<int[], Set<int[]>> inputEdges = new TreeMap<>(Ints.lexicographicalComparator()); //key: vertex. Values: vertices that the key vertex receives input from
            Map<int[], Set<int[]>> outputEdges = new TreeMap<>(Ints.lexicographicalComparator()); //key: vertex. Values: vertices that the key vertex outputs to


            for (int i : vertices) {
                int[] key = {i};
                if (getVertexInDegree(key) < 1) {
                    noIncoming.add(key);
                }

                List<Edge<OpState>> edges = getEdgesOut(new int[]{i});
                Set<int[]> outVertices = new TreeSet<>(Ints.lexicographicalComparator());
                Set<int[]> currInputs = new TreeSet<>(Ints.lexicographicalComparator());
                for (Edge<OpState> edge : edges) {
                    outVertices.add(edge.getTo());
                    Set<int[]> outputSetForInputIdx = outputEdges.get(new int[]{i});
                    if (outputSetForInputIdx == null) {
                        outputSetForInputIdx = new TreeSet<>(Ints.lexicographicalComparator());
                        outputEdges.put(new int[]{i}, outputSetForInputIdx);
                    }

                    outputSetForInputIdx.add(edge.getTo()); //input vertex outputs to the current vertex
                }

                if( getIncomingEdges().get(new int[]{i}) != null) {
                    for (Edge<OpState> edge : getIncomingEdges().get(new int[]{i})) {
                        currInputs.add(edge.getFrom());

                    }

                    inputEdges.put(new int[]{i}, currInputs);
                }
                else
                    inputEdges.put(new int[]{i}, currInputs);

            }


            while (!noIncoming.isEmpty()) {
                int[] next = noIncoming.removeFirst();
                retList.addAll(Ints.asList(next));
                List<int[]> vertexOutputsTo = outputEdges.containsKey(next) ? new ArrayList<>(outputEdges.get(next)) : null;

                //Remove edges next -> vertexOuputsTo[...] from graph;
                if (vertexOutputsTo != null) {
                    //fCollections.sort(vertexOutputsTo);
                    for (int[] v : vertexOutputsTo) {
                        Set<int[]> set = inputEdges.get(v);
                        if (set != null)
                            set.remove(next);
                        if (set == null || set.isEmpty()) {
                            noIncoming.add(v); //No remaining edges for vertex i -> add to list for processing
                        }
                    }
                }
            }

            //If any edges remain in the graph: graph has cycles:
            for (Map.Entry<int[], Set<int[]>> entry : inputEdges.entrySet()) {
                Set<int[]> set = entry.getValue();
                if (set == null)
                    continue;
                if (!set.isEmpty())
                    throw new IllegalStateException("Graph has cycles");
            }

            return Ints.toArray(retList);

        }


        return Ints.toArray(retList);
    }

    /**
     * Topological sort over vertex ids
     * @return
     */
    public int[] topologicalSort() {
        return topologicalSort(false);
    }
}
