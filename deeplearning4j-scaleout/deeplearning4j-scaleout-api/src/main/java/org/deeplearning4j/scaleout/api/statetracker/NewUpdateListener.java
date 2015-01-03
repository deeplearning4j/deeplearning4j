package org.deeplearning4j.scaleout.api.statetracker;

import java.io.Serializable;

/**
 * A new update (used as a hook in to the state tracker for external concerns)
 * @author Adam Gibson
 */
public interface NewUpdateListener {


    /**
     * Update handler
     * @param update the update
     */
    void onUpdate(Serializable update);

}
