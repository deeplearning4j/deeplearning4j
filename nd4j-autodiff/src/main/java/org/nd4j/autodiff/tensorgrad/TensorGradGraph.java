package org.nd4j.autodiff.tensorgrad;

import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpExecAction;
import org.nd4j.autodiff.opstate.OpExecOrder;
import org.nd4j.autodiff.opstate.OpState;

import java.util.*;

/**
 * Created by agibsonccc on 4/11/17.
 */
public class TensorGradGraph extends Graph<NDArrayInformation,OpState> {

    public List<NDArrayInformation> getInputs() {
        List<NDArrayInformation> ret = new ArrayList<>();
        for(int i = 0; i < numVertices(); i++) {
            if(getVertexDegree(i) < 1)
                ret.add(getVertex(i).getValue());
        }

        return ret;
    }



    public TensorGradGraph optimize() {
        OpExecOrder opOrder = getOpOrder();

        for(OpExecAction opExecAction : opOrder.getActions()) {

        }
        return this;
    }

    public OpExecOrder getOpOrder() {
        int[] order = topologicalSort();
        List<OpExecAction> ret = new ArrayList<>();
        //iterate over op execution order skipping
        // nodes that are only inputs
        //the goal is to get all of the needed op executions
        for(int i = 0; i < order.length; i++) {
            //skip vertices that are only inputs
            if(getVertexInDegree(i) < 1) {
                continue;
            }

            int numInputs = Math.max(1,getVertexInDegree(i));
            int inputsCount = 0;
            NDArrayInformation[] inputs = new NDArrayInformation[numInputs];
            int[] inputIds = new int[numInputs];
            List<Edge<OpState>> inputOpStates = getIncomingEdges().get(i);
            //get the inputs for this this output array
            for(Edge<OpState> edge : inputOpStates) {
                if(edge.getTo() == i) {
                    inputs[inputsCount] = getInformationFor(edge.getFrom());
                    inputIds[inputsCount] = edge.getFrom();
                    inputsCount++;
                }
            }

            //add edges
            for(Edge<OpState> opStateEdge : inputOpStates) {
                ret.add(OpExecAction.builder()
                        .output(opStateEdge.getValue().getResult())
                        .opState(opStateEdge.getValue())
                        .inputs(inputs)
                        .inputsIds(inputIds)
                        .outputId(i)
                        .build());
            }


        }

        return OpExecOrder.builder().actions(ret).build();
    }

    public NDArrayInformation getInformationFor(int vertex) {
        return getVertex(vertex).getValue();
    }


    public int[] topologicalSort() {
        LinkedList<Integer> noIncoming = new LinkedList<>();
        Map<Integer, Set<Integer>> inputEdges = new HashMap<>(); //key: vertex. Values: vertices that the key vertex receives input from
        Map<Integer, Set<Integer>> outputEdges = new HashMap<>(); //key: vertex. Values: vertices that the key vertex outputs to
        int[] ret = new int[numVertices()];
        for(int i = 0; i < numVertices(); i++) {
            if(getVertexInDegree(i) < 1) {
                noIncoming.add(i);
            }
            else
                continue;

            List<Edge<OpState>> edges = getEdgesOut(i);
            Set<Integer> outVertices = new HashSet<>();
            for(Edge<OpState> edge : edges) {
                outVertices.add(edge.getTo());
                Set<Integer> outputSetForInputIdx = outputEdges.get(i);
                if (outputSetForInputIdx == null) {
                    outputSetForInputIdx = new HashSet<>();
                    outputEdges.put(i, outputSetForInputIdx);
                }

                outputSetForInputIdx.add(edge.getTo()); //input vertex outputs to the current vertex
            }

            inputEdges.put(i,outVertices);
        }



        int outCounter = 0;
        while(!noIncoming.isEmpty() && outCounter < ret.length) {
            int next = noIncoming.removeFirst();
            ret[outCounter++] = next;
            Set<Integer> vertexOutputsTo = outputEdges.get(next);
            //Remove edges next -> vertexOuputsTo[...] from graph;
            if (vertexOutputsTo != null) {
                for (Integer v : vertexOutputsTo) {
                    Set<Integer> set = inputEdges.get(v);
                    if(set != null)
                        set.remove(next);
                    if (set == null || set.isEmpty()) {
                        noIncoming.add(v); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }


        return ret;
    }


}
