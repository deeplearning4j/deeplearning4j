package org.deeplearning4j.nn.transferlearning;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.graph.vertex.impl.SubsetVertex;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.*;

/**
 * This class is intended for use with the transfer learning API.
 * Often times transfer learning models have "frozen" layers where parameters are held a constant during training
 * For ease of training and quick turn around times, the dataset to be trained on can be featurized and saved to disk.
 * Featurizing in this case refers to conducting a forward pass on the network and saving the activations from the output
 * of the frozen layers. During training the forward pass and the backward pass through the frozen layers can be skipped entirely.
 */
public class TransferLearningHelper extends TransferLearning{

    private boolean isGraph = false;
    private ComputationGraph origGraph;
    private MultiLayerNetwork origMLN;
    private ComputationGraph unFrozenSubsetGraph;
    private MultiLayerNetwork unFrozenSubsetMLN;
    Set<String> frozenInputVertices = new HashSet<>();

    /**
     * Expecting a computation graph or a multilayer network with frozen layer/vertices
     * @param orig either a computation graph or a multi layer network
     */
    public TransferLearningHelper(Model orig) {
        if (orig instanceof ComputationGraph) {
            isGraph = true;
            origGraph = (ComputationGraph) orig;
            initHelperGraph();
        }
        else if (orig instanceof MultiLayerNetwork) {
            origMLN = (MultiLayerNetwork) orig;
            initHelperMLN();
        }
        else {
            throw new IllegalArgumentException("Unknown model.");
        }
    }

    public ComputationGraph unfrozenGraph() {
        return unFrozenSubsetGraph;
    }

    /**
     * Runs through the comp graph and saves off a new model that is simply the "unfrozen" part of the origModel
     * This "unfrozen" model is then used for training and featurizing
     */
    private void initHelperGraph() {

        //parent vertices added in when seen
        Set<String> seenAsParents = new HashSet<>();
        int [] backPropOrder = origGraph.topologicalSortOrder();
        ArrayUtils.reverse(backPropOrder);

        for (int i = 0; i<backPropOrder.length;i++) {
            GraphVertex currentVertex = origGraph.getVertices()[backPropOrder[i]];
            String currentName = currentVertex.getVertexName();
            if (!currentVertex.hasLayer()) {
                //before skipping over a subset vertex check if it has a frozen parent
                if (currentVertex instanceof SubsetVertex) {
                    //if this is a subset vertex and has a parent that is a frozen layer that has not been seen
                    //add to list of frozen inputs
                    VertexIndices[] parentVertices = currentVertex.getInputVertices();
                    for(int j=0; j<parentVertices.length; j++ ) {
                        int parentVertexIndex = parentVertices[j].getVertexIndex();
                        GraphVertex parentVertex = origGraph.getVertices()[parentVertexIndex];
                        if (parentVertex.hasLayer()) {
                            String parentName = origGraph.getVertices()[parentVertexIndex].getVertexName();
                            if (parentVertex.getLayer() instanceof FrozenLayer && !seenAsParents.contains(parentName)) {
                                frozenInputVertices.add(parentName);
                            }
                        }
                    }
                }
            }
            Layer currentLayer = currentVertex.getLayer();
            if (currentLayer instanceof FrozenLayer) {
                //a frozen layer is encountered - should be removed (along with it's inputs)
                //The question is does it need to be an input to the new smaller unfrozen model or not?
                if (!seenAsParents.contains(currentName)) {
                    //not a parent of vertices already seen so needs to be added to the set of inputs
                    frozenInputVertices.add(currentName);
                }
                seenAsParents.add(currentName);
                VertexIndices[] parentVertices = currentVertex.getInputVertices();
                //add parents of current frozen vertex to list of seen parents
                for(int j=0; j<parentVertices.length; j++ ) {
                    int parentVertexIndex = parentVertices[j].getVertexIndex();
                    String parentName = origGraph.getVertices()[parentVertexIndex].getVertexName();
                    seenAsParents.add(parentName);
                }
            }
        }

        TransferLearning.GraphBuilder builder = new TransferLearning.GraphBuilder(origGraph);
        for (String toRemove: seenAsParents) {
            if (frozenInputVertices.contains(toRemove)) {
                builder.removeVertexKeepConnections(toRemove);
            }
            else {
                builder.removeVertexAndConnections(toRemove);
            }
        }

        Set<String> frozenInputVerticesSorted = new HashSet<>();
        frozenInputVerticesSorted.addAll(origGraph.getConfiguration().getNetworkInputs());
        frozenInputVerticesSorted.removeAll(seenAsParents);
        //remove input vertices - just to add back in a predictable order
        for (String existingInput: frozenInputVerticesSorted) {
            builder.removeVertexKeepConnections(existingInput);
        }
        frozenInputVerticesSorted.addAll(frozenInputVertices);
        //to have a predictable order
        List<String> finalInputs = new ArrayList(frozenInputVerticesSorted);
        Collections.sort(finalInputs);
        for (String asInput: frozenInputVerticesSorted) {
            builder.addInputs(asInput);
        }
        unFrozenSubsetGraph = builder.build();

        if (frozenInputVertices.isEmpty()) {
            throw new IllegalArgumentException("No frozen layers found");
        }

    }

    /**
     * Runs through the mln and saves off a new model that is simply the unfrozen part of the origModel
     * This "unfrozen" model is then used for training and featurizing
     */
    private void initHelperMLN() {

        //make smaller graph - loop back in topographical order
        //find a non frozen vertex that has a frozen parent which will always be a layer vertex

        //outputs are the same
        //remove some input vertices
        //set new inputs


    }

}
