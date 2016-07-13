package org.deeplearning4j.gym;

import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.BoxSpace;
import org.deeplearning4j.gym.space.DiscreteSpace;
import org.deeplearning4j.gym.space.ObservationSpace;
import org.json.JSONObject;


/**
 * Created by rubenfiszel on 7/8/16.
 */
public class ClientFactory {

    public static <O, A, OS extends ObservationSpace<O>, AS extends ActionSpace<A>> Client<O, A, OS, AS> build(String url, String envId) {

        JSONObject body = new JSONObject().put("env_id", envId);
        JSONObject reply = ClientUtils.post(url + Client.ENVS_ROOT, body).getObject();

        String instanceId = reply.getString("instance_id");

        OS observationSpace = fetchObservationSpace(url, instanceId);
        AS actionSpace = fetchActionSpace(url, instanceId);

        return new Client(url, envId, instanceId, observationSpace, actionSpace);

    }

    public static <O, A, OS extends ObservationSpace<O>, AS extends ActionSpace<A>>  Client<O, A, OS, AS> build(String envId) {
        return build("http://127.0.0.1:5000", envId);
    }

    public static <AS extends ActionSpace> AS fetchActionSpace(String url, String instanceId) {

        JSONObject reply = ClientUtils.get(url + Client.ENVS_ROOT + instanceId + Client.ACTION_SPACE);
        JSONObject info = reply.getJSONObject("info");
        String infoName = info.getString("name");

        switch (infoName) {
            case "Discrete":
                return (AS) new DiscreteSpace(info.getInt("n"));
            default:
                throw new RuntimeException("Unknown space " + infoName);
        }
    }

    public static <OS extends ObservationSpace> OS fetchObservationSpace(String url, String instanceId) {
        JSONObject reply = ClientUtils.get(url + Client.ENVS_ROOT + instanceId + Client.OBSERVATION_SPACE);
        JSONObject info = reply.getJSONObject("info");
        String infoName = info.getString("name");
        switch (infoName) {
            case "Box":
                return (OS) new BoxSpace(info);
            default:
                throw new RuntimeException("Unknown space " + infoName);
        }
    }
}
