package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * This class is intended for use with the transfer learning API.
 * Often times transfer learning models have "frozen" layers where parameters are held a constant during training
 * For ease of training and quick turn around times, the dataset to be trained on can be featurized and saved to disk.
 * Featurizing in this case refers to conducting a forward pass on the network and saving the activations from the output
 * of the frozen layers. During training the forward pass and the backward pass through the frozen layers can be skipped entirely.
 */
public class TransferLearningHelper extends TransferLearning{

    private boolean isGraph = false;
    private Model origModel;

    /**
     * Expecting a computation graph or a multilayer network with frozen layer/vertices
     * @param orig either a computation graph or a multi layer network
     */
    public TransferLearningHelper(Model orig) {
        origModel = orig;
        if (orig instanceof ComputationGraph) {
            isGraph = true;
            initHelperGraph();
        }
        else {
            initHelperMLN();
        }

    }

    /**
     * Runs through the comp graph and saves off a new model that is simply the "unfrozen" part of the origModel
     * This "unfrozen" model is then used for training and featurizing
     */
    private void initHelperGraph() {

        //loop through layers in order figure out which layer is frozen

        //make smaller model with unfrozen layers (new layer names etc)
        //map small layers to orig layers

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
