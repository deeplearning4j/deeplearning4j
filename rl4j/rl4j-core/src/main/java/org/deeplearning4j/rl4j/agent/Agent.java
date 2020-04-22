package org.deeplearning4j.rl4j.agent;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.agent.listener.AgentListenerList;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.nd4j.base.Preconditions;

import java.util.Map;

public class Agent<ACTION> {
    @Getter
    private final String id;

    @Getter
    private final Environment<ACTION> environment;

    @Getter
    private final IPolicy<ACTION> policy;

    private final TransformProcess transformProcess;

    protected final AgentListenerList<ACTION> listeners;

    private final Integer maxEpisodeSteps;

    @Getter(AccessLevel.PROTECTED)
    private Observation observation;

    @Getter(AccessLevel.PROTECTED)
    private ACTION lastAction;

    @Getter
    private int episodeStepNumber;

    @Getter
    private double reward;

    protected boolean canContinue;

    private Agent(Builder<ACTION> builder) {
        this.environment = builder.environment;
        this.transformProcess = builder.transformProcess;
        this.policy = builder.policy;
        this.maxEpisodeSteps = builder.maxEpisodeSteps;
        this.id = builder.id;

        listeners = buildListenerList();
    }

    protected AgentListenerList<ACTION> buildListenerList() {
        return new AgentListenerList<ACTION>();
    }

    public void addListener(AgentListener listener) {
        listeners.add(listener);
    }

    public void run() {
        runEpisode();
    }

    protected void onBeforeEpisode() {
        // Do Nothing
    }

    protected void onAfterEpisode() {
        // Do Nothing
    }

    protected void runEpisode() {
        reset();
        onBeforeEpisode();

        canContinue = listeners.notifyBeforeEpisode(this);

        while (canContinue && !environment.isEpisodeFinished() && (maxEpisodeSteps == null || episodeStepNumber < maxEpisodeSteps)) {
            performStep();
        }

        if(!canContinue) {
            return;
        }

        onAfterEpisode();
    }

    protected void reset() {
        resetEnvironment();
        resetPolicy();
        reward = 0;
        lastAction = getInitialAction();
        canContinue = true;
    }

    protected void resetEnvironment() {
        episodeStepNumber = 0;
        Map<String, Object> channelsData = environment.reset();
        this.observation = transformProcess.transform(channelsData, episodeStepNumber, false);
    }

    protected void resetPolicy() {
        policy.reset();
    }

    protected ACTION getInitialAction() {
        return environment.getSchema().getActionSchema().getNoOp();
    }

    protected void performStep() {

        onBeforeStep();

        ACTION action = decideAction(observation);

        canContinue = listeners.notifyBeforeStep(this, observation, action);
        if(!canContinue) {
            return;
        }

        StepResult stepResult = act(action);
        handleStepResult(stepResult);

        onAfterStep(stepResult);

        canContinue = listeners.notifyAfterStep(this, stepResult);
        if(!canContinue) {
            return;
        }

        incrementEpisodeStepNumber();
    }

    protected void incrementEpisodeStepNumber() {
        ++episodeStepNumber;
    }

    protected ACTION decideAction(Observation observation) {
        if (!observation.isSkipped()) {
            lastAction = policy.nextAction(observation);
        }

        return lastAction;
    }

    protected StepResult act(ACTION action) {
        return environment.step(action);
    }

    protected void handleStepResult(StepResult stepResult) {
        observation = convertChannelDataToObservation(stepResult, episodeStepNumber + 1);
        reward +=computeReward(stepResult);
    }

    protected Observation convertChannelDataToObservation(StepResult stepResult, int episodeStepNumberOfObs) {
        return transformProcess.transform(stepResult.getChannelsData(), episodeStepNumberOfObs, stepResult.isTerminal());
    }

    protected double computeReward(StepResult stepResult) {
        return stepResult.getReward();
    }

    protected void onAfterStep(StepResult stepResult) {
        // Do Nothing
    }

    protected void onBeforeStep() {
        // Do Nothing
    }

    public static <ACTION> Builder<ACTION> builder(@NonNull Environment<ACTION> environment, @NonNull TransformProcess transformProcess, @NonNull IPolicy<ACTION> policy) {
        return new Builder<>(environment, transformProcess, policy);
    }

    public static class Builder<ACTION> {
        private final Environment<ACTION> environment;
        private final TransformProcess transformProcess;
        private final IPolicy<ACTION> policy;
        private Integer maxEpisodeSteps = null; // Default, no max
        private String id;

        public Builder(@NonNull Environment<ACTION> environment, @NonNull TransformProcess transformProcess, @NonNull IPolicy<ACTION> policy) {
            this.environment = environment;
            this.transformProcess = transformProcess;
            this.policy = policy;
        }

        public Builder<ACTION> maxEpisodeSteps(int maxEpisodeSteps) {
            Preconditions.checkArgument(maxEpisodeSteps > 0, "maxEpisodeSteps must be greater than 0, got", maxEpisodeSteps);
            this.maxEpisodeSteps = maxEpisodeSteps;

            return this;
        }

        public Builder<ACTION> id(String id) {
            this.id = id;
            return this;
        }

        public Agent build() {
            return new Agent(this);
        }
    }
}