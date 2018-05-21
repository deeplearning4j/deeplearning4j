package org.deeplearning4j.malmo;

import org.deeplearning4j.rl4j.space.ObservationSpace;

import com.microsoft.msr.malmo.WorldState;

/**
 * Abstract base class for all Malmo-specific observation spaces
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public abstract class MalmoObservationSpace implements ObservationSpace<MalmoBox> {
    public abstract MalmoBox getObservation(WorldState world_state);
}
