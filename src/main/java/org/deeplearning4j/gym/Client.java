package org.deeplearning4j.gym;


import org.deeplearning4j.gym.space.ActionSpace;
import org.deeplearning4j.gym.space.BoxSpace;
import org.deeplearning4j.gym.space.DiscreteSpace;
import org.deeplearning4j.gym.space.ObservationSpace;
import org.json.JSONObject;

import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 * Created by rubenfiszel on 7/6/16.
 */
public class Client<O, A> {


    public static String V1_ROOT = "/v1";
    public static String ENVS_ROOT = V1_ROOT + "/envs/";

    public static String MONITOR_START = "/monitor/start/";
    public static String MONITOR_CLOSE = "/monitor/close/";
    public static String CLOSE = "/close/";
    public static String RESET = "/reset/";
    public static String SHUTDOWN = "/shutdown/";
    public static String UPLOAD = "/upload/";
    public static String STEP = "/step/";
    public static String OBSERVATION_SPACE = "/observation_space/";
    public static String ACTION_SPACE = "/action_space/";

    private String envId;
    private String instanceId;
    private String url;
    private ObservationSpace<O> observationSpace;
    private ActionSpace<A> actionSpace;

    public Client(String url, String envId, String instanceId, ObservationSpace<O> observationSpace, ActionSpace<A> actionSpace) {

        this.envId = envId;
        this.url = url;
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace;
        this.instanceId = instanceId;

    }

    public static ActionSpace getActionSpace(String url, String instanceId) {

        JSONObject reply = ClientUtils.get(url + Client.ENVS_ROOT + instanceId + ACTION_SPACE);
        JSONObject info = reply.getJSONObject("info");
        String infoName = info.getString("name");

        switch (infoName) {
            case "Discrete":
                return new DiscreteSpace(info.getInt("n"));
            default:
                throw new RuntimeException("Unknown space " + infoName);
        }
    }

    public static ObservationSpace getObservationSpace(String url, String instanceId) {
        JSONObject reply = ClientUtils.get(url + Client.ENVS_ROOT + instanceId + OBSERVATION_SPACE);
        JSONObject info = reply.getJSONObject("info");
        String infoName = info.getString("name");
        switch (infoName) {
            case "Box":
                return new BoxSpace(info);
            default:
                throw new RuntimeException("Unknown space " + infoName);
        }
    }

    public static Set<String> listAll(String url) {
        JSONObject reply = ClientUtils.get(url + ENVS_ROOT);
        return reply.getJSONObject("envs").keySet();
    }

    public static void serverShutdown(String url) {
        ClientUtils.post(url + ENVS_ROOT + SHUTDOWN, new JSONObject());
    }

    public String getInstanceId() {
        return instanceId;
    }

    public String getEnvId() {
        return envId;
    }

    public String getUrl() {
        return url;
    }

    public ObservationSpace<O> getObservationSpace() {
        return observationSpace;
    }

    public ActionSpace<A> getActionSpace() {
        return actionSpace;
    }

    public Set<String> listAll() {
        return listAll(url);
    }

    public void reset() {
        ClientUtils.post(url + ENVS_ROOT + instanceId + RESET, new JSONObject());
    }

    public void monitorStart(String directory, boolean force, boolean resume) {
        JSONObject json = new JSONObject()
                .put("directory", directory)
                .put("force", force)
                .put("resume", resume);

        ClientUtils.post(url + ENVS_ROOT + instanceId + MONITOR_START, json);
    }

    public void monitorStart(String directory) {
        JSONObject json = new JSONObject()
                .put("directory", directory);

        ClientUtils.post(url + ENVS_ROOT + instanceId + MONITOR_START, json);
    }

    public void monitorClose() {
        ClientUtils.post(url + ENVS_ROOT + instanceId + MONITOR_CLOSE, new JSONObject());
    }

    public void close() {
        ClientUtils.post(url + ENVS_ROOT + instanceId + CLOSE, new JSONObject());
    }

    public void upload(String trainingDir, String apiKey, String algorithmId) {
        JSONObject json = new JSONObject()
                .put("training_dir", trainingDir)
                .put("api_key", apiKey)
                .put("algorithm_id", algorithmId);

        ClientUtils.post(url + ENVS_ROOT + instanceId + CLOSE, json);
    }

    public void upload(String trainingDir, String apiKey) {
        JSONObject json = new JSONObject()
                .put("training_dir", trainingDir)
                .put("api_key", apiKey);
        try {
            ClientUtils.post(url + V1_ROOT + instanceId + UPLOAD, json);
        } catch (RuntimeException e) {
            Logger logger = Logger.getLogger("Client Upload");
            logger.log(Level.SEVERE, "Impossible to upload: Wrong API key?");
        }
    }

    public void ServerShutdown() {
        serverShutdown(url);
    }

    public StepReply<O> step(A action) {
        JSONObject body = new JSONObject()
                .put("action", action);

        JSONObject reply = ClientUtils.post(url + ENVS_ROOT + instanceId + STEP, body).getObject();

        O observation = observationSpace.getValue(reply, "observation");
        double reward = reply.getDouble("reward");
        boolean done = reply.getBoolean("done");
        JSONObject info = reply.getJSONObject("info");

        return new StepReply<O>(observation,  reward, done, info);
    }


}
