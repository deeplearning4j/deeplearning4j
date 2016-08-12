package org.deeplearning4j.rl4j.mdp.gym;


import org.deeplearning4j.rl4j.Client;
import org.deeplearning4j.rl4j.ClientFactory;
import org.deeplearning4j.rl4j.StepReply;
import org.deeplearning4j.rl4j.gym.space.HighLowDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 */
public class GymEnv<O, A, AS extends ActionSpace<A>> implements MDP<O, A, AS> {

    final public static String GYM_MONITOR_DIR = "/tmp/gym-dqn";

    final private Client<O, A, AS> client;
    final private String envId;
    final private boolean render;
    private ActionTransformer actionTransformer = null;
    private boolean done = false;

    public GymEnv(String envId, boolean render) {
        this.client = ClientFactory.build(envId, render);
        this.envId = envId;
        this.render = render;
        client.monitorStart(GYM_MONITOR_DIR, true, false);
    }

    public GymEnv(String envId, boolean render, int[] actions) {
        this(envId, render);
        actionTransformer = new ActionTransformer((HighLowDiscrete) getActionSpace(), actions);
    }


    public ObservationSpace<O> getObservationSpace() {
        return client.getObservationSpace();
    }

    public AS getActionSpace() {
        if (actionTransformer == null)
            return client.getActionSpace();
        else
            return (AS) actionTransformer;
    }

    public StepReply<O> step(A action) {
        StepReply<O> stepRep = client.step(action);
        done = stepRep.isDone();
        return stepRep;
    }

    public boolean isDone() {
        return done;
    }

    public O reset() {
        done = false;
        return client.reset();
    }


    public void upload(String apiKey) {
        client.upload(GYM_MONITOR_DIR, apiKey);
    }

    public void close() {
        client.monitorClose();
    }

    public GymEnv<O, A, AS> newInstance() {
        return new GymEnv<O, A, AS>(envId, render);
    }
}
