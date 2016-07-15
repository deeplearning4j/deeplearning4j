package org.deeplearning4j.gym.space;

import java.util.Random;

/**
 * Created by rubenfiszel on 7/8/16.
 */
public class DiscreteSpace implements ActionSpace<Integer> {

    private int n;
    private Random rd;

    public DiscreteSpace(int n) {

        this.n = n;
        rd = new Random();
    }

    public int getN() {
        return n;
    }

    public Integer randomAction() {
        return rd.nextInt(n);
    }

}
