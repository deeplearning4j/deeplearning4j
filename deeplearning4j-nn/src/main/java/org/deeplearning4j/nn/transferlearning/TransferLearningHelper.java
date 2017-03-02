package org.deeplearning4j.nn.transferlearning;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.graph.vertex.impl.SubsetVertex;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.*;

/**
 * This class is intended for use with the transfer learning API.
 * Often times transfer learning models have "frozen" layers where parameters are held constant during training
 * For ease of training and quick turn around times, the dataset to be trained on can be featurized and saved to disk.
 * Featurizing in this case refers to conducting a forward pass on the network and saving the activations from the output
 * of the frozen layers.
 * During training the forward pass and the backward pass through the frozen layers can be skipped entirely and the "featurized"
 * dataset can be fit with the smaller unfrozen part of the computation graph which allows for quicker iterations.
 * The class internally traverses the computation graph/MLN and builds an instance of the computation graph/MLN that is
 * equivalent to the unfrozen subset.
 *
 * @author susaneraly
 */
public class TransferLearningHelper {

    private boolean isGraph = true;
    private ComputationGraph origGraph;
    private MultiLayerNetwork origMLN;
    private ComputationGraph unFrozenSubsetGraph;
    private MultiLayerNetwork unFrozenSubsetMLN;
    Set<String> frozenInputVertices = new HashSet<>(); //name map so no problem
    List<String> graphInputs;
    int frozenInputLayer = 0;

    /**
     * Expecting a computation graph or a multilayer network with frozen layer/vertices
     *
     * @param orig either a computation graph or a multi layer network
     */
    public TransferLearningHelper(ComputationGraph orig) {
        origGraph = orig;
        initHelperGraph();
    }

    public TransferLearningHelper(MultiLayerNetwork orig) {
        isGraph = false;
        origMLN = orig;
        initHelperMLN();
    }

    public void errorIfGraphIfMLN() {
        if (isGraph)
            throw new IllegalArgumentException("This instance was initialized with a computation graph. Cannot apply methods related to MLN");
        else
            throw new IllegalArgumentException("This instance was initialized with a MultiLayerNetwork. Cannot apply methods related to computation graphs");

    }

    /**
     * Returns the unfrozen subset of the computation graph
     * Note that with each call to featurizedFit the parameters to the original computation graph are also updated
     */
    protected ComputationGraph unfrozenGraph() {
        if (!isGraph) errorIfGraphIfMLN();
        return unFrozenSubsetGraph;
    }

    /**
     * Returns the unfrozen layers of the MultiLayerNetwork
     * Note that with each call to featurizedFit the parameters to the original MLN are also updated
     */
    protected MultiLayerNetwork unfrozenMLN() {
        if (isGraph) errorIfGraphIfMLN();
        return unFrozenSubsetMLN;
    }

    /**
     * Runs through the comp graph and saves off a new model that is simply the "unfrozen" part of the origModel
     * This "unfrozen" model is then used for training with featurized data
     */
    private void initHelperGraph() {

        //parent vertices added in when seen
        Set<String> seenAsParents = new HashSet<>();
        int[] backPropOrder = origGraph.topologicalSortOrder().clone();
        ArrayUtils.reverse(backPropOrder);

        for (int i = 0; i < backPropOrder.length; i++) {
            GraphVertex currentVertex = origGraph.getVertices()[backPropOrder[i]];
            String currentName = currentVertex.getVertexName();
            if (!currentVertex.hasLayer()) {
                //before skipping over a subset vertex check if it has a frozen parent
                if (currentVertex instanceof SubsetVertex) {
                    //if this is a subset vertex and has a parent that is a frozen layer that has not been seen
                    //add to list of frozen inputs
                    VertexIndices[] parentVertices = currentVertex.getInputVertices();
                    for (int j = 0; j < parentVertices.length; j++) {
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
                for (int j = 0; j < parentVertices.length; j++) {
                    int parentVertexIndex = parentVertices[j].getVertexIndex();
                    String parentName = origGraph.getVertices()[parentVertexIndex].getVertexName();
                    seenAsParents.add(parentName);
                }
            }
        }

        TransferLearning.GraphBuilder builder = new TransferLearning.GraphBuilder(origGraph);
        for (String toRemove : seenAsParents) {
            if (frozenInputVertices.contains(toRemove)) {
                builder.removeVertexKeepConnections(toRemove);
            } else {
                builder.removeVertexAndConnections(toRemove);
            }
        }

        Set<String> frozenInputVerticesSorted = new HashSet<>();
        frozenInputVerticesSorted.addAll(origGraph.getConfiguration().getNetworkInputs());
        frozenInputVerticesSorted.removeAll(seenAsParents);
        //remove input vertices - just to add back in a predictable order
        for (String existingInput : frozenInputVerticesSorted) {
            builder.removeVertexKeepConnections(existingInput);
        }
        frozenInputVerticesSorted.addAll(frozenInputVertices);
        //Sort all inputs to the computation graph - in order to have a predictable order
        graphInputs = new ArrayList(frozenInputVerticesSorted);
        Collections.sort(graphInputs);
        for (String asInput : frozenInputVerticesSorted) {
            //add back in the right order
            builder.addInputs(asInput);
        }
        unFrozenSubsetGraph = builder.build();
        copyOrigParamsToSubsetGraph();
        unFrozenSubsetGraph.setListeners(origGraph.getListeners());

        if (frozenInputVertices.isEmpty()) {
            throw new IllegalArgumentException("No frozen layers found");
        }

    }

    private void initHelperMLN() {
        for (int i = 0; i < origMLN.getnLayers(); i++) {
            if (origMLN.getLayer(i) instanceof FrozenLayer) {
                frozenInputLayer = i;
            }
        }
        List<NeuralNetConfiguration> allConfs = new ArrayList<>();
        for (int i = frozenInputLayer + 1; i < origMLN.getnLayers(); i++) {
            allConfs.add(origMLN.getLayer(i).conf());
        }

        MultiLayerConfiguration c = origMLN.getLayerWiseConfigurations();

        unFrozenSubsetMLN = new MultiLayerNetwork(new MultiLayerConfiguration.Builder()
                .backprop(c.isBackprop())
                .inputPreProcessors(c.getInputPreProcessors())
                .pretrain(c.isPretrain())
                .backpropType(c.getBackpropType())
                .tBPTTForwardLength(c.getTbpttFwdLength())
                .tBPTTBackwardLength(c.getTbpttBackLength())
                .confs(allConfs).build());
        unFrozenSubsetMLN.init();
        //copy over params
        for (int i = frozenInputLayer + 1; i < origMLN.getnLayers(); i++) {
            unFrozenSubsetMLN.getLayer(i - frozenInputLayer - 1).setParams(origMLN.getLayer(i).params());
        }
        unFrozenSubsetMLN.setListeners(origMLN.getListeners());
    }

    /**
     * During training frozen vertices/layers can be treated as "featurizing" the input
     * The forward pass through these frozen layer/vertices can be done in advance and the dataset saved to disk to iterate
     * quickly on the smaller unfrozen part of the model
     * Currently does not support datasets with feature masks
     *
     * @param input multidataset to feed into the computation graph with frozen layer vertices
     * @return a multidataset with input features that are the outputs of the frozen layer vertices and the original labels.
     */
    public MultiDataSet featurize(MultiDataSet input) {
        if (!isGraph) {
            throw new IllegalArgumentException("Cannot use multidatasets with MultiLayerNetworks.");
        }
        INDArray[] labels = input.getLabels();
        INDArray[] features = input.getFeatures();
        if (input.getFeaturesMaskArrays() != null) {
            throw new IllegalArgumentException("Currently cannot support featurizing datasets with feature masks");
        }
        INDArray[] featureMasks = null;
        INDArray[] labelMasks = input.getLabelsMaskArrays();

        INDArray[] featuresNow = new INDArray[graphInputs.size()];
        Map<String, INDArray> activationsNow = origGraph.feedForward(features, false);
        for (int i = 0; i < graphInputs.size(); i++) {
            String anInput = graphInputs.get(i);
            if (origGraph.getVertex(anInput).isInputVertex()) {
                //was an original input to the graph
                int inputIndex = origGraph.getConfiguration().getNetworkInputs().indexOf(anInput);
                featuresNow[i] = origGraph.getInput(inputIndex);
            } else {
                //needs to be grabbed from the internal activations
                featuresNow[i] = activationsNow.get(anInput);
            }
        }

        return new MultiDataSet(featuresNow, labels, featureMasks, labelMasks);
    }

    /**
     * During training frozen vertices/layers can be treated as "featurizing" the input
     * The forward pass through these frozen layer/vertices can be done in advance and the dataset saved to disk to iterate
     * quickly on the smaller unfrozen part of the model
     * Currently does not support datasets with feature masks
     *
     * @param input multidataset to feed into the computation graph with frozen layer vertices
     * @return a multidataset with input features that are the outputs of the frozen layer vertices and the original labels.
     */
    public DataSet featurize(DataSet input) {
        if (isGraph) {
            //trying to featurize for a computation graph
            if (origGraph.getNumInputArrays() > 1 || origGraph.getNumOutputArrays() > 1) {
                throw new IllegalArgumentException("Input or output size to a computation graph is greater than one. Requires use of a MultiDataSet.");
            } else {
                if (input.getFeaturesMaskArray() != null) {
                    throw new IllegalArgumentException("Currently cannot support featurizing datasets with feature masks");
                }
                MultiDataSet inbW = new MultiDataSet(
                        new INDArray[]{input.getFeatures()}, new INDArray[]{input.getLabels()},
                        null, new INDArray[]{input.getLabelsMaskArray()});
                MultiDataSet ret = featurize(inbW);
                return new DataSet(ret.getFeatures()[0], input.getLabels(), ret.getLabelsMaskArrays()[0], input.getLabelsMaskArray());
            }
        } else {
            if (input.getFeaturesMaskArray() != null)
                throw new UnsupportedOperationException("Feature masks not supported with featurizing currently");
            return new DataSet(origMLN.feedForwardToLayer(frozenInputLayer + 1, input.getFeatures(), false)
                    .get(frozenInputLayer + 1), input.getLabels(), null, input.getLabelsMaskArray());
        }
    }

    /**
     * Fit from a featurized dataset.
     * The fit is conducted on an internally instantiated subset model that is representative of the unfrozen part of the original model.
     * After each call on fit the parameters for the original model are updated
     *
     * @param iter
     */
    public void fitFeaturized(MultiDataSetIterator iter) {
        unFrozenSubsetGraph.fit(iter);
        copyParamsFromSubsetGraphToOrig();
    }

    public void fitFeaturized(MultiDataSet input) {
        unFrozenSubsetGraph.fit(input);
        copyParamsFromSubsetGraphToOrig();
    }

    public void fitFeaturized(DataSet input) {
        if (isGraph) {
            unFrozenSubsetGraph.fit(input);
            copyParamsFromSubsetGraphToOrig();
        } else {
            unFrozenSubsetMLN.fit(input);
            copyParamsFromSubsetMLNToOrig();
        }
    }

    public void fitFeaturized(DataSetIterator iter) {
        if (isGraph) {
            unFrozenSubsetGraph.fit(iter);
            copyParamsFromSubsetGraphToOrig();
        } else {
            unFrozenSubsetMLN.fit(iter);
            copyParamsFromSubsetMLNToOrig();
        }
    }

    private void copyParamsFromSubsetGraphToOrig() {
        for (GraphVertex aVertex : unFrozenSubsetGraph.getVertices()) {
            if (!aVertex.hasLayer()) continue;
            origGraph.getVertex(aVertex.getVertexName()).getLayer().setParams(aVertex.getLayer().params());
        }
    }

    private void copyOrigParamsToSubsetGraph() {
        for (GraphVertex aVertex : unFrozenSubsetGraph.getVertices()) {
            if (!aVertex.hasLayer()) continue;
            aVertex.getLayer().setParams(origGraph.getLayer(aVertex.getVertexName()).params());
        }
    }

    private void copyParamsFromSubsetMLNToOrig() {
        for (int i = frozenInputLayer + 1; i < origMLN.getnLayers(); i++) {
            origMLN.getLayer(i).setParams(unFrozenSubsetMLN.getLayer(i - frozenInputLayer - 1).params());
        }
    }

}
