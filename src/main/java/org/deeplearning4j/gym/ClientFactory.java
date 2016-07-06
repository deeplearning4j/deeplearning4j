package org.deeplearning4j.gym;

import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.ObservationSpace;
import org.json.JSONObject;


/**
 * Created by rubenfiszel on 7/8/16.
 */
public class ClientFactory {

    public static <O, A> Client<O, A> build(String url, String envId) {

        JSONObject body = new JSONObject().put("env_id", envId);
        JSONObject reply = ClientUtils.post(url + Client.ENVS_ROOT, body).getObject();

        String instanceId = reply.getString("instance_id");

        ObservationSpace<O> observationSpace = Client.getObservationSpace(url, instanceId);
        ActionSpace<A> actionSpace = Client.getActionSpace(url, instanceId);

        return new Client<O, A>(url, envId, instanceId, observationSpace, actionSpace);

    }

    public static <O, A> Client<O, A> build(String envId) {
        return build("http://127.0.0.1:5000", envId);
    }
}
