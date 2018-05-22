package org.deeplearning4j.rl4j.mdp.toy;

import lombok.Value;
import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 */
@Value
public class SimpleToyState implements Encodable {

    int i;
    int step;

    @Override
    public double[] toArray() {
        double[] ar = new double[1];
        ar[0] = (20 - i);
        return ar;
    }

}
