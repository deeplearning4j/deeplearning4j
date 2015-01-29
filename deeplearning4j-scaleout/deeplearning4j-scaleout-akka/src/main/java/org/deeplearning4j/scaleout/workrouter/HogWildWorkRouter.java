package org.deeplearning4j.scaleout.workrouter;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.api.workrouter.BaseWorkRouter;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;

/**
 *
 * Async updates
 *
 * @author Adam Gibson
 */
public class HogWildWorkRouter extends BaseWorkRouter  {
    public HogWildWorkRouter() {
    }

    public HogWildWorkRouter(StateTracker stateTracker) {
        super(stateTracker);
    }

    @Override
    public StateTracker createStateTracker(Configuration conf) throws Exception {
        return new HazelCastStateTracker();
    }

    @Override
    public boolean sendWork() {
        return true;
    }


}
