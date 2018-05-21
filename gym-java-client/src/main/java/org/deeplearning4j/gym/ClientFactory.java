package org.deeplearning4j.gym;

import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.GymObservationSpace;
import org.deeplearning4j.rl4j.space.HighLowDiscrete;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *
 * ClientFactory contains builder method to create a new {@link Client}
 */
public class ClientFactory {

    public static <O, A, AS extends ActionSpace<A>> Client<O, A, AS> build(String url, String envId, boolean render) {

        JSONObject body = new JSONObject().put("env_id", envId);
        JSONObject reply = ClientUtils.post(url + Client.ENVS_ROOT, body).getObject();

        String instanceId;

        try {
            instanceId = reply.getString("instance_id");
        } catch (JSONException e) {
            throw new RuntimeException("Environment id not found", e);
        }

        GymObservationSpace<O> observationSpace = fetchObservationSpace(url, instanceId);
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
            case "HighLow":
                int numRows = info.getInt("num_rows");
                int size = 3 * numRows;
                JSONArray matrixJson = info.getJSONArray("matrix");
                INDArray matrix = Nd4j.create(numRows, 3);
                for (int i = 0; i < size; i++) {
                    matrix.putScalar(i, matrixJson.getDouble(i));
                }
                matrix.reshape(numRows, 3);
                return (AS) new HighLowDiscrete(matrix);
            default:
                throw new RuntimeException("Unknown action space " + infoName);
        }
    }

    public static <O> GymObservationSpace<O> fetchObservationSpace(String url, String instanceId) {
        JSONObject reply = ClientUtils.get(url + Client.ENVS_ROOT + instanceId + Client.OBSERVATION_SPACE);
        JSONObject info = reply.getJSONObject("info");
        return new GymObservationSpace<O>(info);
    }
}
