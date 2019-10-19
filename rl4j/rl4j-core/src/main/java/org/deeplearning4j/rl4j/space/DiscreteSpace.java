package org.deeplearning4j.rl4j.space;

import lombok.Getter;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *         <p>
 *         A discrete space of action. A discrete space is always isomorphic
 *         to a space of integer so we can parametrize directly by Integer.
 *         Benefit of using Integers directly is that you can use it as the
 *         id of the node assigned to that action in the outpout of a DQN.
 */
public class DiscreteSpace implements ActionSpace<Integer> {

    //size of the space also defined as the number of different actions
    @Getter
    final protected int size;

    protected Random rnd;

    public DiscreteSpace(int size) {
        this(size, Nd4j.getRandom());
    }

    public DiscreteSpace(int size, Random rnd) {
        this.size = size;
        this.rnd = rnd;
    }

    public Integer randomAction() {
        return rnd.nextInt(size);
    }

    public Object encode(Integer a) {
        return a;
    }

    public Integer noOp() {
        return 0;
    }

    public void setRandom(Random rnd) {
        this.rnd = rnd;
    }
}
