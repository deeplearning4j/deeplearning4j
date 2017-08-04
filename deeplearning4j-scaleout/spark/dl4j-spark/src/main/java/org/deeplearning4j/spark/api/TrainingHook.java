package org.deeplearning4j.spark.api;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.Serializable;

/**
 * A hook for the workers when training.
 * A pre update and post update method are specified
 * for when certain information needs to be collected
 * or there needs to be specific parameters
 * or models sent to remote locations for visualization
 * or other things.
 *
 * @author Adam Gibson
 */
public interface TrainingHook extends Serializable {
    /**
     * A hook method for pre update.
     * @param minibatch the inibatch
     *                  that was used for the update
     * @param model themodel that was update
     */
    void preUpdate(DataSet minibatch, Model model);

    /**
     * A hook method for post update
     * @param minibatch the minibatch
     *                  that was usd for the update
     * @param model the model that was updated
     */
    void postUpdate(DataSet minibatch, Model model);

    /**
     * A hook method for pre update.
     * @param minibatch the inibatch
     *                  that was used for the update
     * @param model the model that was update
     */
    void preUpdate(MultiDataSet minibatch, Model model);

    /**
     * A hook method for post update
     * @param minibatch the minibatch
     *                  that was usd for the update
     * @param model the model that was updated
     */
    void postUpdate(MultiDataSet minibatch, Model model);

}
