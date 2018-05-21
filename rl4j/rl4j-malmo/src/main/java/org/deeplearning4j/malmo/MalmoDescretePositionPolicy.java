package org.deeplearning4j.malmo;

import java.util.Arrays;

import com.microsoft.msr.malmo.WorldState;

/**
 * A Malmo consistency policy that ensures the both there is a reward and next observation has a different position that the previous one.
 * This will only work for your mission if you require that every action moves to a new location. 
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoDescretePositionPolicy implements MalmoObservationPolicy {
    MalmoObservationSpacePosition observationSpace = new MalmoObservationSpacePosition();

    @Override
    public boolean isObservationConsistant(WorldState world_state, WorldState original_world_state) {
        MalmoBox last_observation = observationSpace.getObservation(world_state);
        MalmoBox old_observation = observationSpace.getObservation(original_world_state);

        double[] newvalues = old_observation == null ? null : old_observation.toArray();
        double[] oldvalues = last_observation == null ? null : last_observation.toArray();

        return !(world_state.getObservations().isEmpty() || world_state.getRewards().isEmpty()
                        || Arrays.equals(oldvalues, newvalues));
    }

}
