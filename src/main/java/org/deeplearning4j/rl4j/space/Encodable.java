package org.deeplearning4j.rl4j.space;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/19/16.
 *         Encodable is an interface that ensure that the state is convertible to a double array
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
