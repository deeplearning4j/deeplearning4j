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

package org.deeplearning4j.malmo;

import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

import com.microsoft.msr.malmo.AgentHost;
import com.microsoft.msr.malmo.ClientInfo;
import com.microsoft.msr.malmo.ClientPool;
import com.microsoft.msr.malmo.MissionRecordSpec;
import com.microsoft.msr.malmo.MissionSpec;
import com.microsoft.msr.malmo.TimestampedReward;
import com.microsoft.msr.malmo.WorldState;

import lombok.Setter;
import lombok.Getter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * MDP Wrapper around Malmo Java Client Library
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoEnv implements MDP<MalmoBox, Integer, DiscreteSpace> {
    // Malmo Java client library depends on native library
    static {
        String malmoHome = System.getenv("MALMO_HOME");

        if (malmoHome == null)
            throw new RuntimeException("MALMO_HOME must be set to your Malmo environement.");

        try {
            if (Files.exists(Paths.get(malmoHome + "/Java_Examples/libMalmoJava.jnilib"))) {
            	System.load(malmoHome + "/Java_Examples/libMalmoJava.jnilib");
            } else if (Files.exists(Paths.get(malmoHome + "/Java_Examples/MalmoJava.dll"))) {
            	System.load(malmoHome + "/Java_Examples/MalmoJava.dll");
            } else if (Files.exists(Paths.get(malmoHome + "/Java_Examples/libMalmoJava.so"))) {
            	System.load(malmoHome + "/Java_Examples/libMalmoJava.so");
            } else {
                System.load("MalmoJava");
            }
        } catch( UnsatisfiedLinkError e ) {
        	throw new RuntimeException("MALMO_HOME must be set to your Malmo environement. Could not load native library at '" + malmoHome + "/Java_Examples/'", e );
        }
    }

    final static private int NUM_RETRIES = 10;

    final private Logger logger;

    @Setter
    MissionSpec mission = null;

    MissionRecordSpec missionRecord = null;
    ClientPool clientPool = null;

    @Getter
    MalmoObservationSpace observationSpace;

    @Getter
    MalmoActionSpace actionSpace;

    MalmoObservationPolicy framePolicy;

    AgentHost agent_host = null;
    WorldState last_world_state;
    MalmoBox last_observation;

    @Setter
    MalmoResetHandler resetHandler = null;

    /**
     * Create a MalmoEnv using a XML-file mission description and minimal set of parameters.
     * Equivalent to MalmoEnv( loadMissionXML( missionFileName ), missionRecord, actionSpace, observationSpace, framePolicy, clientPool )
     * @param missionFileName Name of XML file describing mission
     * @param actionSpace Malmo action space implementation
     * @param observationSpace Malmo observation space implementation
     * @param framePolicy Malmo frame policy implementation
     * @param clientPool Malmo client pool in host:port string form. If set to null, will default to a single Malmo client at 127.0.0.1:10000
     */
    public MalmoEnv(String missionFileName, MalmoActionSpace actionSpace, MalmoObservationSpace observationSpace,
                    MalmoObservationPolicy framePolicy, String... clientPool) {
        this(loadMissionXML(missionFileName), null, actionSpace, observationSpace, framePolicy, null);

        if (clientPool == null || clientPool.length == 0)
            clientPool = new String[] {"127.0.0.1:10000"};
        this.clientPool = new ClientPool();
        for (String client : clientPool) {
            String parts[] = client.split(":");
            this.clientPool.add(new ClientInfo(parts[0], Short.parseShort(parts[1])));
        }
    }

    /**
     * Create a fully specified MalmoEnv
     * @param mission Malmo mission specification.
     * @param missionRecord Malmo record specification. Ignored if set to NULL
     * @param actionSpace Malmo action space implementation
     * @param observationSpace Malmo observation space implementation
     * @param framePolicy Malmo frame policy implementation
     * @param clientPool Malmo client pool; cannot be null
     */
    public MalmoEnv(MissionSpec mission, MissionRecordSpec missionRecord, MalmoActionSpace actionSpace,
                    MalmoObservationSpace observationSpace, MalmoObservationPolicy framePolicy, ClientPool clientPool) {
        this.mission = mission;
        this.missionRecord = missionRecord != null ? missionRecord : new MissionRecordSpec();
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
        this.framePolicy = framePolicy;
        this.clientPool = clientPool;

        logger = LoggerFactory.getLogger(this.getClass());
    }

    /**
     * Convenience method to load a Malmo mission specification from an XML-file
     * @param filename name of XML file
     * @return Mission specification loaded from XML-file
     */
    public static MissionSpec loadMissionXML(String filename) {
        MissionSpec mission = null;
        try {
            String xml = new String(Files.readAllBytes(Paths.get(filename)));
            mission = new MissionSpec(xml, true);
        } catch (Exception e) {
            //e.printStackTrace();
            throw new RuntimeException( e );
        }

        return mission;
    }

    @Override
    public MalmoBox reset() {
        close();

        if (resetHandler != null) {
            resetHandler.onReset(this);
        }

        agent_host = new AgentHost();

        int i;
        for (i = NUM_RETRIES - 1; i > 0; --i) {
            try {
                Thread.sleep(100 + 500 * (NUM_RETRIES - i - 1));
                agent_host.startMission(mission, clientPool, missionRecord, 0, "rl4j_0");
            } catch (Exception e) {
                logger.warn("Error starting mission: " + e.getMessage() + " Retrying " + i + " more times.");
                continue;
            }
            break;
        }

        if (i == 0) {
            close();
            throw new MalmoConnectionError("Unable to connect to client.");
        }

        logger.info("Waiting for the mission to start");

        do {
            last_world_state = agent_host.getWorldState();
        } while (!last_world_state.getIsMissionRunning());

        //We need to make sure all the blocks are in place before we grab the observations
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
        }

        last_world_state = waitForObservations(false);

        last_observation = observationSpace.getObservation(last_world_state);

        return last_observation;
    }

    private WorldState waitForObservations(boolean andRewards) {
        WorldState world_state;
        WorldState original_world_state = last_world_state;
        boolean missingData;

        do {
            Thread.yield();
            world_state = agent_host.peekWorldState();
            missingData = world_state.getObservations().isEmpty() || world_state.getVideoFrames().isEmpty();
            missingData = andRewards
                            ? missingData || !framePolicy.isObservationConsistant(world_state, original_world_state)
                            : missingData;
        } while (missingData && world_state.getIsMissionRunning());

        return agent_host.getWorldState();
    }

    @Override
    public void close() {
        if (agent_host != null) {
            agent_host.delete();
        }

        agent_host = null;
    }

    @Override
    public StepReply<MalmoBox> step(Integer action) {
        agent_host.sendCommand((String) actionSpace.encode(action));

        last_world_state = waitForObservations(true);
        last_observation = observationSpace.getObservation(last_world_state);

        if (isDone()) {
            logger.info("Mission ended");
        }

        return new StepReply<MalmoBox>(last_observation, getRewards(last_world_state), isDone(), null);
    }

    private double getRewards(WorldState world_state) {
        double rval = 0;

        for (int i = 0; i < world_state.getRewards().size(); i++) {
            TimestampedReward reward = world_state.getRewards().get(i);
            rval += reward.getValue();
        }

        return rval;
    }

    @Override
    public boolean isDone() {
        return !last_world_state.getIsMissionRunning();
    }

    @Override
    public MDP<MalmoBox, Integer, DiscreteSpace> newInstance() {
        MalmoEnv rval = new MalmoEnv(mission, missionRecord, actionSpace, observationSpace, framePolicy, clientPool);
        rval.setResetHandler(resetHandler);
        return rval;
    }
}
