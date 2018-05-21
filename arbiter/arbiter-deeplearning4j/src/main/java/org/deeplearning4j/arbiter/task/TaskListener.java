package org.deeplearning4j.arbiter.task;

import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

/**
 * TaskListener: can be used to preprocess and post process a model (MultiLayerNetwork or ComputationGraph) before/after
 * training, in a {@link MultiLayerNetworkTaskCreator} or {@link ComputationGraphTaskCreator}
 *
 * @author Alex Black
 */
public interface TaskListener extends Serializable {

    /**
     * Preprocess the model, before any training has taken place.
     * <br>
     * Can be used to (for example) set listeners on a model before training starts
     * @param model     Model to preprocess
     * @param candidate Candidate information, for the current model
     * @return The updated model (usually the same one as the input, perhaps with modifications)
     */
    <T extends Model> T preProcess(T model, Candidate candidate);

    /**
     * Post process the model, after any training has taken place
     * @param model     Model to postprocess
     * @param candidate Candidate information, for the current model
     */
    void postProcess(Model model, Candidate candidate);

}
