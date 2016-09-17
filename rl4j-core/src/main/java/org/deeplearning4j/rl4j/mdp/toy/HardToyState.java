package org.deeplearning4j.rl4j.mdp.toy;

import lombok.Value;
import org.deeplearning4j.gym.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 */
@Value
public class HardToyState implements Encodable {

    double[] values;
    int step;

    public double[] toArray() {
        return values;
    }
}
