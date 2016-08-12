package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.rl4j.StepReply;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 */
public abstract class QLearningDiscrete<O extends Encodable> extends QLearning<O, Integer, DiscreteSpace> {

    @Getter
    final private QLConfiguration configuration;
    @Getter
    final private DataManager dataManager;
    @Getter
    final private MDP<O, Integer, DiscreteSpace> mdp;
    @Getter
    private DQNPolicy<O> policy;
    @Getter
    private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;
    @Getter
    final private IDQN currentDQN;
    @Getter
    @Setter
    private IDQN targetDQN;
    private int lastAction;
    private INDArray history = null;
    private int lastMonitor = -Constants.MONITOR_FREQ;


    public QLearningDiscrete(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf, DataManager dataManager) {
        super(conf);
        this.configuration = conf;
        this.mdp = mdp;
        this.dataManager = dataManager;
        currentDQN = dqn;
        targetDQN = dqn.clone();
        policy = new DQNPolicy(getCurrentDQN());
        egPolicy = new EpsGreedy(policy, mdp, conf.getUpdateStart(), conf.getEpsilonDecreaseRate(), getRandom(), conf.getMinEpsilon(), this);


    }


    public void postEpoch() {


        if (getHistoryProcessor() != null && getHistoryProcessor().isMonitoring())
            getHistoryProcessor().stopMonitor();

    }

    public void preEpoch() {
        history = null;

        if (getStepCounter() - lastMonitor >= Constants.MONITOR_FREQ && getHistoryProcessor() != null) {
            lastMonitor = getStepCounter();
            getHistoryProcessor().startMonitor(getDataManager().getVideoDir() + "/video-" + getEpochCounter() + "-" + getStepCounter() + ".mp4");
        }
    }

    protected QLStepReturn<O> trainStep(O obs) {

        Integer action;
        INDArray input = getInput(obs);
        boolean isHistoryProcessor = getHistoryProcessor() != null;

        if (isHistoryProcessor)
            getHistoryProcessor().record(input);

        int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;
        int historyLength = isHistoryProcessor ? getHistoryProcessor().getConf().getHistoryLength() : 1;
        int updateStart = getConfiguration().getUpdateStart()+((getConfiguration().getBatchSize()+historyLength)*skipFrame);

        Double maxQ = Double.NaN; //ignore if Nan for stats

        if (getStepCounter() % skipFrame != 0) {
            action = lastAction;
        } else {
            if (history == null) {
                if (isHistoryProcessor) {
                    getHistoryProcessor().add(input);
                    history = getHistoryProcessor().getHistory();
                } else
                    history = input;
            }
            INDArray qs = getCurrentDQN().output(history);
            int maxAction = Learning.getMaxAction(qs);

           // System.out.println("MAX ACTION: " +maxAction + " " + qs + " " + qs.shapeInfoToString());

            maxQ = qs.getDouble(maxAction);
            action = getEgPolicy().nextAction(history);
        }
        lastAction = action;

        StepReply<O> stepReply = getMdp().step(action);

        if (isHistoryProcessor)
            getHistoryProcessor().add(getInput(stepReply.getObservation()));

        INDArray nhistory = isHistoryProcessor ? getHistoryProcessor().getHistory() : getInput(stepReply.getObservation());

        if (getStepCounter() % skipFrame == 0) {

            Transition<Integer> trans = new Transition(history, action, stepReply.getReward(), stepReply.isDone(), nhistory);
            getExpReplay().store(trans);
            if (getStepCounter() > updateStart) {
                Pair<INDArray, INDArray> targets = setTarget(getExpReplay().getBatch());
                getCurrentDQN().fit(targets.getFirst(), targets.getSecond());
            }
        }

        history = nhistory;

        return new QLStepReturn<O>(maxQ, getCurrentDQN().getLatestScore(), stepReply);

    }


    protected Pair<INDArray, INDArray> setTarget(ArrayList<Transition<Integer>> transitions) {
        if (transitions.size() == 0)
            throw new IllegalArgumentException("too few transitions");

        int size = transitions.size();

        int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape() : getHistoryProcessor().getConf().getShape();
        int[] nshape = makeShape(size, shape);
        INDArray obs = Nd4j.create(nshape);
        INDArray nextObs = Nd4j.create(nshape);
        int[] actions = new int[size];
        boolean[] areTerminal = new boolean[size];

        for (int i = 0; i < size; i++) {
            Transition<Integer> trans = transitions.get(i);
            areTerminal[i] = trans.isTerminal();
            actions[i] = trans.getAction();
            obs.putRow(i, trans.getObservation());
            nextObs.putRow(i, trans.getNextObservation());
        }

        INDArray dqnOutputAr = dqnOutput(obs);
        //System.out.println("OLD" + dqnOutputAr);
        INDArray dqnOutputNext = dqnOutput(nextObs);
        INDArray targetDqnOutputNext = null;

        INDArray tempQ = null;
        INDArray getMaxAction = null;
        if (getConfiguration().isDoubleDQN()) {
            targetDqnOutputNext = targetDqnOutput(nextObs);
            getMaxAction = Nd4j.argMax(dqnOutputNext, 1);
        } else {
            tempQ = Nd4j.max(dqnOutputNext, 1);
        }


        //System.out.println("BEFORE  II: " +obs.getDouble(0, 0) + " " + dqnOutput.getDouble(0, 0) + " " + dqnOutput.getDouble(0, 1));
        //INDArray test = Nd4j.create(new float[]{0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f}).reshape(10, 1);
        //System.out.println("TEST:" + dqnOutput(test));


        for (int i = 0; i < size; i++) {
            double yTar = transitions.get(i).getReward();
            // if (i==0)  System.out.println(yTar);
            if (!areTerminal[i]) {
                double q = 0;
                if (getConfiguration().isDoubleDQN()) {
                    q += targetDqnOutputNext.getDouble(i, getMaxAction.getInt(i));
                } else
                    q += tempQ.getDouble(i);
                yTar += getConfiguration().getGamma() * q;
                //  if (i==0) System.out.println("AFTER: " +yTar);
                //if (i == 0)
                //System.out.println(yErr + " " + q + " " + transitions.get(i).getReward() + " " );
            }
            //System.out.println(i + " " + maxAction);
            double previousV = dqnOutputAr.getDouble(i, actions[i]);
            double lowB = previousV - getConfiguration().getErrorClamp();
            double highB = previousV + getConfiguration().getErrorClamp();
            double clamped = Math.min(highB, Math.max(yTar, lowB));
            //System.out.println("CLAMP: " + previousV + " " + clamped);
            dqnOutputAr.putScalar(i, actions[i], clamped);
        }
        //for (int i = 0; i < size; i++) {
        //System.out.println(i + " II: " +obs.getDouble(i) + " " + dqnOutput.getDouble(i));
        //}
        //System.out.println("AFTER  II: " +obs.getDouble(0, 0) + " " + dqnOutput.getDouble(0, 0) + " " + dqnOutput.getDouble(0, 1) + " " +dqnOutput.getDouble(1));

        return new Pair(obs, dqnOutputAr);
    }

}
