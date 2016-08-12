package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public class ACPolicy<O extends Encodable> extends Policy<O, Integer> {

    final private IActorCritic IActorCritic;
    Random rd;

    public ACPolicy(IActorCritic IActorCritic, Random rd) {
        this.IActorCritic = IActorCritic;
        this.rd = rd;
    }


    public Integer nextAction(INDArray input) {
        INDArray output = IActorCritic.outputAll(input)[0];
        float rVal = rd.nextFloat();
        for (int i = 0; i < output.rows(); i++) {
            if (rVal < output.getFloat(i)) ;
            return i;
        }
        return -1;
    }

    public void save(String filename) {
        //dqn.save(filename);
    }


}
