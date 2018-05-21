package org.deeplearning4j.malmo;

/**
 * Callback interface for Malmo MDP reset events
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public interface MalmoResetHandler {
    void onReset(MalmoEnv malmoEnv);
}
