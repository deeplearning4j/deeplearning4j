package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.val;
import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.*;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.collection.IntArrayKeySet;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.*;

/**
 * Graph data structure for tensors
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Data
public class SDGraph extends Graph<SDVariable,OpState> {

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
                    Map<Integer, Vertex<SDVariable>> vertices,
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
    public void addVertex(Vertex<SDVariable> ndArrayInformationVertex) {
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
    public List<int[]> getOutputIds() {
        List<int[]> ret = new ArrayList<>();
        for (int i : getVertices().keySet()) {
            if (getEdgesOut(new int[]{i}).size() < 1)
                ret.add(new int[]{i});
        }

        return ret;
    }



    /**
     * Get the input vertices
     * @return
     */
    public List<SDVariable> getInputs() {
        List<SDVariable> ret = new ArrayList<>();
        List<SDVariable> outVars = sameDiff.variables();
        for (SDVariable entry : outVars) {
            if(numInputsFor(entry.getVertexId()) < 1)
                ret.add(entry);
        }

        return ret;
    }

    /**
     * Get the number of inputs for a particular vertex id
     * @param vertexId the vertex id to check
     * @return the number of inputs for a particular vertex
     */
    public int numInputsFor(int...vertexId)  {
        if(getIncomingEdges().containsKey(vertexId))
            return getIncomingEdges().get(vertexId).size();
        return 0;
    }

    /**
     *
     * @return
     */
    public OpExecOrder getOpOrder(boolean reverse) {
        int[][] order = topologicalSort(reverse);
        Set<OpState> seenStates = new HashSet<>();
        if(reverse) {
            List<OpExecAction> forwardActions = getOpOrder().getActions();
            Map<int[],OpExecAction> opExecActionMap = new IntArrayKeyMap<>();
            //size the vertex id relative to where it would be encountered
            //in a reverse order traversal
            final Map<int[],Integer> normalForwardOrderMap = new IntArrayKeyMap<>();
            for(int i = forwardActions.size() - 1,j = 0; i >= 0; i--,j++) {
                OpExecAction currAction = forwardActions.get(i);
                normalForwardOrderMap.put(currAction.getOutputId(),j);
                opExecActionMap.put(currAction.getOutputId(),currAction);
            }


            Set<Edge<OpState>> allEdges = new HashSet<>();
            Collection<List<Edge<OpState>>> outgoingEdges = getEdges().values();
            for(List<Edge<OpState>> edge : outgoingEdges) {
                allEdges.addAll(edge);
            }



            PriorityQueue<int[]> depthQueue = new PriorityQueue<>(allEdges.size(), new Comparator<int[]>() {
                @Override
                public int compare(int[] o1, int[] o2) {
                    int o1MaxDepth = getMaxDepth(o1);
                    int o2MaxDepth = getMaxDepth(o2);
                    return Ints.compare(-o1MaxDepth,-o2MaxDepth);
                }
            });


            List<IntArrayKeyMap.IntArray> vertices = new ArrayList<>();
            for(List<Edge<OpState>> edge : getEdges().values())  {
                for(Edge<OpState> edge1 : edge) {
                    if(!vertices.contains(new IntArrayKeyMap.IntArray(edge1.getTo()))) {
                        vertices.add(new IntArrayKeyMap.IntArray(edge1.getTo()));
                    }
                }
            }

            Collections.sort(vertices, new Comparator<IntArrayKeyMap.IntArray>() {
                @Override
                public int compare(IntArrayKeyMap.IntArray ints, IntArrayKeyMap.IntArray t1) {
                    return Ints.compare(Ints.max(ints.getBackingArray()),Ints.max(t1.getBackingArray()));
                }
            });



            for(IntArrayKeyMap.IntArray i : vertices) {
                depthQueue.add(i.getBackingArray());

            }



            List<OpExecAction> ret = new ArrayList<>();
            while(!depthQueue.isEmpty()) {
                int[] ndArrayVertex = depthQueue.poll();
                OpExecAction action = opExecActionMap.get(ndArrayVertex);
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
                if (getVertexInDegree(order[i]) < 1) {
                    continue;
                }

                int numInputs = Math.max(1, getVertexInDegree(order[i]));
                int inputsCount = 0;
                List<Integer> inputIdsList = new ArrayList<>();
                List<Edge<OpState>> inputOpStates = getIncomingEdges().get(order[i]);
                List<SDVariable> inputInfo = new ArrayList<>();
                //get the inputs for this this output array
                for (Edge<OpState> edge : inputOpStates) {
                    inputIdsList.addAll(Ints.asList(edge.getFrom()));
                    for(int input : edge.getFrom())  {
                        Preconditions.checkNotNull(getVariableForVertex(input));
                        inputInfo.add(getVariableForVertex(input));
                        inputsCount++;
                    }
                }

                // Preconditions.checkState(inputsCount == numInputs, "Not all inputs were filled.");
                //add edges
                Edge<OpState> opStateEdge = inputOpStates.get(0);
                if(!seenStates.contains(opStateEdge.getValue())) {
                    ret.add(OpExecAction.builder()
                            .opState(opStateEdge.getValue())
                            .inputsIds(Ints.toArray(inputIdsList))
                            .outputId(order[i])
                            .build());
                    seenStates.add(opStateEdge.getValue());
                }
            }


            Collections.sort(ret, new Comparator<OpExecAction>() {
                @Override
                public int compare(OpExecAction o1, OpExecAction o2) {
                    return Ints.compare(Ints.max(o1.getOutputId()),Ints.max(o2.getOutputId()));
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
     * {@link SDVariable}
     * accessor for a given vertex
     * @param vertex the vertex id
     * @return the information for the vertex
     */
    public SDVariable getVariableForVertex(int vertex) {
        Vertex<SDVariable> ndArrayInformation = getVertex(vertex);
        if(ndArrayInformation == null)
            return null;
        return ndArrayInformation.getValue();
    }


    /**
     * Topological sort over vertex ids
     * @return
     */
    public int[][] topologicalSort(boolean reverse) {
        List<int[]> vertices = new ArrayList<>();
        for(List<Edge<OpState>> edge : getEdges().values())  {
            for(Edge<OpState> edge1 : edge) {
                if(!ArrayUtil.listOfIntsContains(vertices,edge1.getTo())) {
                    vertices.add(edge1.getTo());
                }

                if(!ArrayUtil.listOfIntsContains(vertices,edge1.getFrom())) {
                    vertices.add(edge1.getFrom());
                }

            }
        }

        Collections.sort(vertices, new Comparator<int[]>() {
            @Override
            public int compare(int[] ints, int[] t1) {
                return Ints.compare(Ints.max(ints),Ints.max(t1));
            }
        });


        List<int[]> retList = new ArrayList<>();

        if(reverse) {
            List<OpExecAction> forwardActions = getOpOrder().getActions();
            //size the vertex id relative to where it would be encountered
            //in a reverse order traversal
            final Map<int[],Integer> normalForwardOrderMap = new HashMap<>();
            for(int i = forwardActions.size() - 1,j = 0; i >= 0; i--,j++) {
                OpExecAction currAction = forwardActions.get(i);
                normalForwardOrderMap.put(currAction.getOutputId(),j);
            }


            Collections.reverse(vertices);


            Set<Edge<OpState>> allEdges = new HashSet<>();
            Collection<List<Edge<OpState>>> outgoingEdges = getEdges().values();
            for(List<Edge<OpState>> edge : outgoingEdges) {
                allEdges.addAll(edge);
            }


            PriorityQueue<int[]> depthQueue = new PriorityQueue<>(allEdges.size(), new Comparator<int[]>() {
                @Override
                public int compare(int[] o1, int[] o2) {
                    int o1MaxDepth = getMaxDepth(o1);
                    int o2MaxDepth = getMaxDepth(o2);
                    return Ints.compare(-o1MaxDepth,-o2MaxDepth);
                }
            });

            for(int[] i : vertices) {
                depthQueue.add(i);

            }

            while(!depthQueue.isEmpty()) {
                int[] vertex =  depthQueue.poll();
                retList.add(vertex);
            }


        }
        else {
            LinkedList<int[]> noIncoming = new LinkedList<>();
            Map<int[], Set<int[]>> inputEdges = new IntArrayKeyMap<>(); //key: vertex. Values: vertices that the key vertex receives input from
            Map<int[], Set<int[]>> outputEdges = new IntArrayKeyMap<>(); //key: vertex. Values: vertices that the key vertex outputs to


            for (int[] i : vertices) {
                int[] key = i;
                if (getVertexInDegree(key) < 1) {
                    noIncoming.add(key);
                }

                List<Edge<OpState>> edges = getEdgesOut(i);
                Set<int[]> outVertices = new IntArrayKeySet();
                Set<int[]> currInputs = new IntArrayKeySet();
                for (Edge<OpState> edge : edges) {
                    outVertices.add(edge.getTo());
                    Set<int[]> outputSetForInputIdx = outputEdges.get(i);
                    if (outputSetForInputIdx == null) {
                        outputSetForInputIdx = new IntArrayKeySet();
                        outputEdges.put(i, outputSetForInputIdx);
                    }

                    outputSetForInputIdx.add(edge.getTo()); //input vertex outputs to the current vertex
                }

                if( getIncomingEdges().get(i) != null) {
                    for (Edge<OpState> edge : getIncomingEdges().get(i)) {
                        currInputs.add(edge.getFrom());

                    }

                    inputEdges.put(i, currInputs);
                }
                else
                    inputEdges.put(i, currInputs);

            }

            if(noIncoming.isEmpty()) {
                throw new IllegalStateException("No ops found. Unable to execute.");
            }
            while (!noIncoming.isEmpty()) {
                int[] next = noIncoming.removeFirst();
                retList.add(next);
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

            int[][] ret = new int[retList.size()][];
            for(int i = 0; i < retList.size(); i++)
                ret[i] = retList.get(i);
            return ret;

        }


        int[][] ret = new int[retList.size()][];
        for(int i = 0; i < retList.size(); i++)
            ret[i] = retList.get(i);
        return ret;

    }



    public int getMaxDepth(int[] vertexIdx) {
        int ret = -1;
        for(int vertexId : vertexIdx)
            if(getVertex(vertexId).depth() > ret)
                ret = getVertex(vertexId).depth();
        return ret;
    }
    /**
     * Topological sort over vertex ids
     * @return
     */
    public int[][] topologicalSort() {
        return topologicalSort(false);
    }
}
