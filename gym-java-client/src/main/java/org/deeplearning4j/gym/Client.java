/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.gym;


import com.mashape.unirest.http.JsonNode;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.space.GymObservationSpace;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.json.JSONObject;

import java.util.Set;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/6/16.
 *
 * A client represent an active connection to a specific instance of an environment on a rl4j-http-api server.
 * for API specification
 *
 * @param <O>  Observation type
 * @param <A>  Action type
 * @param <AS> Action Space type
 * @see <a href="https://github.com/openai/gym-http-api#api-specification">https://github.com/openai/gym-http-api#api-specification</a>
 */
@Slf4j
@Value
public class Client<O, A, AS extends ActionSpace<A>> {


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


    String url;
    String envId;
    String instanceId;
    GymObservationSpace<O> observationSpace;
    AS actionSpace;
    boolean render;


    /**
     * @param url url of the server
     * @return set of all environments running on the server at the url
     */
    public static Set<String> listAll(String url) {
        JSONObject reply = ClientUtils.get(url + ENVS_ROOT);
        return reply.getJSONObject("envs").keySet();
    }

    /**
     * Shutdown the server at the url
     *
     * @param url url of the server
     */
    public static void serverShutdown(String url) {
        ClientUtils.post(url + ENVS_ROOT + SHUTDOWN, new JSONObject());
    }

    /**
     * @return set of all environments running on the same server than this client
     */
    public Set<String> listAll() {
        return listAll(url);
    }

    /**
     * Step the environment by one action
     *
     * @param action action to step the environment with
     * @return the StepReply containing the next observation, the reward, if it is a terminal state and optional information.
     */
    public StepReply<O> step(A action) {
        JSONObject body = new JSONObject().put("action", getActionSpace().encode(action)).put("render", render);

        JSONObject reply = ClientUtils.post(url + ENVS_ROOT + instanceId + STEP, body).getObject();

        O observation = observationSpace.getValue(reply, "observation");
        double reward = reply.getDouble("reward");
        boolean done = reply.getBoolean("done");
        JSONObject info = reply.getJSONObject("info");

        return new StepReply<O>(observation, reward, done, info);
    }

    /**
     * Reset the state of the environment and return an initial observation.
     *
     * @return initial observation
     */
    public O reset() {
        JsonNode resetRep = ClientUtils.post(url + ENVS_ROOT + instanceId + RESET, new JSONObject());
        return observationSpace.getValue(resetRep.getObject(), "observation");
    }

    /*
    Present in the doc but not working currently server-side
    public void monitorStart(String directory) {
    
        JSONObject json = new JSONObject()
                .put("directory", directory);
    
        monitorStartPost(json);
    }
    */

    /**
     * Start monitoring.
     *
     * @param directory path to directory in which store the monitoring file
     * @param force     clear out existing training data from this directory (by deleting every file prefixed with "openaigym.")
     * @param resume    retain the training data already in this directory, which will be merged with our new data
     */
    public void monitorStart(String directory, boolean force, boolean resume) {
        JSONObject json = new JSONObject().put("directory", directory).put("force", force).put("resume", resume);

        monitorStartPost(json);
    }

    private void monitorStartPost(JSONObject json) {
        ClientUtils.post(url + ENVS_ROOT + instanceId + MONITOR_START, json);
    }

    /**
     * Flush all monitor data to disk
     */
    public void monitorClose() {
        ClientUtils.post(url + ENVS_ROOT + instanceId + MONITOR_CLOSE, new JSONObject());
    }

    /**
     * Upload monitoring data to OpenAI servers.
     *
     * @param trainingDir directory that contains the monitoring data
     * @param apiKey      personal OpenAI API key
     * @param algorithmId an arbitrary string indicating the paricular version of the algorithm (including choices of parameters) you are running.
     **/
    public void upload(String trainingDir, String apiKey, String algorithmId) {
        JSONObject json = new JSONObject().put("training_dir", trainingDir).put("api_key", apiKey).put("algorithm_id",
                        algorithmId);

        uploadPost(json);
    }

    /**
     * Upload monitoring data to OpenAI servers.
     *
     * @param trainingDir directory that contains the monitoring data
     * @param apiKey      personal OpenAI API key
     */
    public void upload(String trainingDir, String apiKey) {
        JSONObject json = new JSONObject().put("training_dir", trainingDir).put("api_key", apiKey);

        uploadPost(json);
    }

    private void uploadPost(JSONObject json) {
        try {
            ClientUtils.post(url + V1_ROOT + UPLOAD, json);
        } catch (RuntimeException e) {
            log.error("Impossible to upload: Wrong API key?");
        }
    }

    /**
     * Shutdown the server at the same url than this client
     */
    public void serverShutdown() {
        serverShutdown(url);
    }

    public ActionSpace getActionSpace(){
        return actionSpace;
    }
}
