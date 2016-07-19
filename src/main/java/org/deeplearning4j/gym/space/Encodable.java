package org.deeplearning4j.gym.space;

/**
 * Created by rubenfiszel on 7/19/16.
 */
public interface Encodable {

    /**
     * $
     * encodes all the information of an Observation in an array double and can be used as input of a DQN directly
     *
     * @return the encoded informations
     */
    double[] toArray();
}
