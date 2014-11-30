package org.deeplearning4j.scaleout.api.workrouter;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.conf.Configuration;

/**
 * Handles job flow
 * @author Adam Gibson
 */
public abstract class BaseWorkRouter implements WorkRouter {

    protected StateTracker stateTracker;
    protected boolean waitForWorkers = true;
    public BaseWorkRouter() {

    }

    public BaseWorkRouter(StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }

    @Override
    public StateTracker stateTracker() {
        return stateTracker;
    }

    @Override
    public void setup(Configuration conf) {
        if(stateTracker == null)
            try {
                stateTracker = createStateTracker(conf);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        waitForWorkers = conf.getBoolean(WAIT_FOR_WORKERS,true);
    }

    @Override
    public boolean isWaitForWorkers() {
        return waitForWorkers;
    }

    public abstract StateTracker createStateTracker(Configuration conf) throws Exception;


}
