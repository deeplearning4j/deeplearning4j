package org.deeplearning4j.gym;

import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.DiscreteSpace;
import org.deeplearning4j.gym.space.ObservationSpace;
import org.json.JSONObject;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *
 * ClientFactory contains builder method to create a new {@link Client}
 */
public class ClientFactory {

    public static <O, A, AS extends ActionSpace<A>> Client<O, A, AS> build(String url, String envId, boolean render) {

        JSONObject body = new JSONObject().put("env_id", envId);
        JSONObject reply = ClientUtils.post(url + Client.ENVS_ROOT, body).getObject();

        String instanceId = reply.getString("instance_id");

        ObservationSpace<O> observationSpace = fetchObservationSpace(url, instanceId);
        AS actionSpace = fetchActionSpace(url, instanceId);

        return new Client(url, envId, instanceId, observationSpace, actionSpace, render);

    }

    public static <O, A, AS extends ActionSpace<A>> Client<O, A, AS> build(String envId, boolean render) {
        return build("http://127.0.0.1:5000", envId, render);
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

    public static <O> ObservationSpace<O> fetchObservationSpace(String url, String instanceId) {
        JSONObject reply = ClientUtils.get(url + Client.ENVS_ROOT + instanceId + Client.OBSERVATION_SPACE);
        JSONObject info = reply.getJSONObject("info");
        return new ObservationSpace<O>(info);
    }
}
