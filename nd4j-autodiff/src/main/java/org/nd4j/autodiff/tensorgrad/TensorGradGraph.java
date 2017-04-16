package org.nd4j.autodiff.tensorgrad;

import org.nd4j.autodiff.graph.Graph;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ops.Op;

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



    public List<OpState> getOpOrder() {
        int[] order = topologicalSort();
        List<OpState> ret = new ArrayList<>();
        for(int i = 0; i < order.length; i++) {
            List<Edge<OpState>> edges = getEdgesOut(i);
            for(Edge<OpState> opStateEdge : edges) {
                ret.add(opStateEdge.getValue());
            }

        }

        return ret;
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

            List<Edge<OpState>> edges = getEdgesOut(i);
            Set<Integer> outVertices = new HashSet<>();
            for(Edge<OpState> edge : edges) {
                outVertices.add(edge.getTo());
                Set<Integer> outputSetForInputIdx = outputEdges.get(i);
                if (outputSetForInputIdx == null) {
                    outputSetForInputIdx = new HashSet<>();
                    outputEdges.put(i, outputSetForInputIdx);
                }

                outputSetForInputIdx.add(i); //input vertex outputs to the current vertex
            }

            inputEdges.put(i,outVertices);
        }



        int outCounter = 0;
        while(!noIncoming.isEmpty()) {
            int next = noIncoming.removeFirst();
            ret[outCounter++] = next;
            Set<Integer> vertexOutputsTo = outputEdges.get(next);
            //Remove edges next -> vertexOuputsTo[...] from graph;
            if (vertexOutputsTo != null) {
                for (Integer v : vertexOutputsTo) {
                    Set<Integer> set = inputEdges.get(v);
                    set.remove(next);
                    if (set.isEmpty()) {
                        noIncoming.add(v); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }

        return ret;
    }


}
