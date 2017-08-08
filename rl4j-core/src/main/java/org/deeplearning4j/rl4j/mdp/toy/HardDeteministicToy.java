package org.deeplearning4j.rl4j.mdp.toy;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.logging.Logger;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 * A toy MDP where the agent should find the maximum to get the reward.
 * Useful to debug as it's very fast to run
 */
public class HardDeteministicToy implements MDP<HardToyState, Integer, DiscreteSpace> {

    final private static int MAX_STEP = 20;
    final private static int SEED = 1234;
    final private static int ACTION_SIZE = 10;
    final private static HardToyState[] states = genToyStates(MAX_STEP, SEED);
    @Getter
    private DiscreteSpace actionSpace = new DiscreteSpace(ACTION_SIZE);
    @Getter
    private ObservationSpace<HardToyState> observationSpace = new ArrayObservationSpace(new int[] {ACTION_SIZE});
    private HardToyState hardToyState;

    public static void printTest(IDQN idqn) {
        INDArray input = Nd4j.create(MAX_STEP, ACTION_SIZE);
        for (int i = 0; i < MAX_STEP; i++) {
            input.putRow(i, Nd4j.create(states[i].toArray()));
        }
        INDArray output = Nd4j.max(idqn.output(input), 1);
        Logger.getAnonymousLogger().info(output.toString());
    }

    public static int maxIndex(double[] values) {
        double maxValue = -Double.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static HardToyState[] genToyStates(int size, int seed) {

        Random rd = new Random(seed);
        HardToyState[] hardToyStates = new HardToyState[size];
        for (int i = 0; i < size; i++) {

            double[] values = new double[ACTION_SIZE];

            for (int j = 0; j < ACTION_SIZE; j++) {
                values[j] = rd.nextDouble();
            }
            hardToyStates[i] = new HardToyState(values, i);
        }

        return hardToyStates;
    }

    public void close() {}

    @Override
    public boolean isDone() {

        return hardToyState.getStep() == MAX_STEP - 1;
    }

    public HardToyState reset() {

        return hardToyState = states[0];
    }

    public StepReply<HardToyState> step(Integer a) {
        double reward = 0;
        if (a == maxIndex(hardToyState.getValues()))
            reward += 1;
        hardToyState = states[hardToyState.getStep() + 1];
        return new StepReply(hardToyState, reward, isDone(), new JSONObject("{}"));
    }

    public HardDeteministicToy newInstance() {
        return new HardDeteministicToy();
    }

}
