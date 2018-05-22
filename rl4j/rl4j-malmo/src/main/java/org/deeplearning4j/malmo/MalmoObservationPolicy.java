package org.deeplearning4j.malmo;

import com.microsoft.msr.malmo.WorldState;

/**
 * A Malmo consistency policy interface.
 * Used by MalmoEnv to ensure next observation is in a consistent state 
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
interface MalmoObservationPolicy {
    boolean isObservationConsistant(WorldState world_state, WorldState original_world_state);
}
