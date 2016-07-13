package org.deeplearning4j.gym;


import org.deeplearning4j.gym.space.BoxSpace;
import org.deeplearning4j.gym.space.DiscreteSpace;

/**
 * Created by rubenfiszel on 7/6/16.
 */
public class ExampleAgent {


    public static void run() {

        Client<Box, Integer, BoxSpace, DiscreteSpace> client = ClientFactory.build("CartPole-v0");

        String outDir = "/tmp/random-agent-results";
        client.monitorStart(outDir, true, false);

        int episodeCount = 1;
        int maxSteps = 200;
        int reward = 0;

        for (int i = 0; i < episodeCount; i++) {
            client.reset();

            for (int j = 0; j < maxSteps; j++) {
                Integer action = client.getActionSpace().randomAction();
                StepReply<Box> step = client.step(action);
                reward += step.getReward();
                if (step.isDone())
                    break;
            }

        }

        client.monitorClose();
        client.upload(outDir,"YOUR_OPENAI_GYM_API_KEY");

    }

    public static void main(String[] args) {
        ExampleAgent.run();
    }

}
