package org.deeplearning4j.gym.test;

import com.mashape.unirest.http.JsonNode;
import org.deeplearning4j.gym.Client;
import org.deeplearning4j.gym.ClientFactory;
import org.deeplearning4j.gym.ClientUtils;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

import static org.mockito.Matchers.eq;
import static org.powermock.api.mockito.PowerMockito.mockStatic;
import static org.powermock.api.mockito.PowerMockito.when;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/11/16.
 */

@RunWith(PowerMockRunner.class)
@PrepareForTest(ClientUtils.class)
public class ClientTest {

    @Test
    public void completeClientTest() {

        String url = "http://127.0.0.1:5000";
        String env = "Powermock-v0";
        String instanceID = "e15739cf";
        String testDir = "/tmp/testDir";
        boolean render = true;
        String renderStr = render ? "True" : "False";

        mockStatic(ClientUtils.class);

        //post mock

        JSONObject buildReq = new JSONObject("{\"env_id\":\"" + env + "\"}");
        JsonNode buildRep = new JsonNode("{\"instance_id\":\"" + instanceID + "\"}");
        when(ClientUtils.post(eq(url + Client.ENVS_ROOT), JSONObjectMatcher.jsonEq(buildReq))).thenReturn(buildRep);

        JSONObject monStartReq = new JSONObject("{\"resume\":false,\"directory\":\"" + testDir + "\",\"force\":true}");
        when(ClientUtils.post(eq(url + Client.ENVS_ROOT + instanceID + Client.MONITOR_START),
                        JSONObjectMatcher.jsonEq(monStartReq))).thenReturn(null);

        JSONObject monStopReq = new JSONObject("{}");
        when(ClientUtils.post(eq(url + Client.ENVS_ROOT + instanceID + Client.MONITOR_CLOSE),
                        JSONObjectMatcher.jsonEq(monStopReq))).thenReturn(null);

        JSONObject resetReq = new JSONObject("{}");
        JsonNode resetRep = new JsonNode(
                        "{\"observation\":[0.021729452941849317,-0.04764548144956857,-0.024914502756611293,-0.04074903379512588]}");
        when(ClientUtils.post(eq(url + Client.ENVS_ROOT + instanceID + Client.RESET),
                        JSONObjectMatcher.jsonEq(resetReq))).thenReturn(resetRep);

        JSONObject stepReq = new JSONObject("{\"action\":0, \"render\":" + renderStr + "}");
        JsonNode stepRep = new JsonNode(
                        "{\"observation\":[0.020776543312857946,-0.24240146656155923,-0.02572948343251381,0.24397017400615437],\"reward\":1,\"done\":false,\"info\":{}}");
        when(ClientUtils.post(eq(url + Client.ENVS_ROOT + instanceID + Client.STEP), JSONObjectMatcher.jsonEq(stepReq)))
                        .thenReturn(stepRep);

        JSONObject stepReq2 = new JSONObject("{\"action\":1, \"render\":" + renderStr + "}");
        JsonNode stepRep2 = new JsonNode(
                        "{\"observation\":[0.020776543312857946,-0.24240146656155923,-0.02572948343251381,0.24397017400615437],\"reward\":1,\"done\":false,\"info\":{}}");
        when(ClientUtils.post(eq(url + Client.ENVS_ROOT + instanceID + Client.STEP),
                        JSONObjectMatcher.jsonEq(stepReq2))).thenReturn(stepRep2);

        //get mock
        JSONObject obsSpace = new JSONObject(
                        "{\"info\":{\"name\":\"Box\",\"shape\":[4],\"high\":[4.8,3.4028234663852886E38,0.41887902047863906,3.4028234663852886E38],\"low\":[-4.8,-3.4028234663852886E38,-0.41887902047863906,-3.4028234663852886E38]}}");
        when(ClientUtils.get(eq(url + Client.ENVS_ROOT + instanceID + Client.OBSERVATION_SPACE))).thenReturn(obsSpace);

        JSONObject actionSpace = new JSONObject("{\"info\":{\"name\":\"Discrete\",\"n\":2}}");
        when(ClientUtils.get(eq(url + Client.ENVS_ROOT + instanceID + Client.ACTION_SPACE))).thenReturn(actionSpace);


        //test

        Client<Box, Integer, DiscreteSpace> client = ClientFactory.build(url, env, render);
        client.monitorStart(testDir, true, false);

        int episodeCount = 1;
        int maxSteps = 200;
        int reward = 0;

        for (int i = 0; i < episodeCount; i++) {
            client.reset();

            for (int j = 0; j < maxSteps; j++) {

                Integer action = ((ActionSpace<Integer>)client.getActionSpace()).randomAction();
                StepReply<Box> step = client.step(action);
                reward += step.getReward();

                //return a isDone true before i == maxSteps
                if (j == maxSteps - 5) {
                    JSONObject stepReqLoc = new JSONObject("{\"action\":0}");
                    JsonNode stepRepLoc = new JsonNode(
                                    "{\"observation\":[0.020776543312857946,-0.24240146656155923,-0.02572948343251381,0.24397017400615437],\"reward\":1,\"done\":true,\"info\":{}}");

                    when(ClientUtils.post(eq(url + Client.ENVS_ROOT + instanceID + Client.STEP),
                                    JSONObjectMatcher.jsonEq(stepReqLoc))).thenReturn(stepRepLoc);
                }

                if (step.isDone()) {
                    //                    System.out.println("break");
                    break;
                }
            }

        }

        client.monitorClose();
        client.upload(testDir, "YOUR_OPENAI_GYM_API_KEY");


    }

}
