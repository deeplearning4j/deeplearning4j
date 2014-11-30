package org.deeplearning4j.scaleout.workrouter;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.api.workrouter.BaseWorkRouter;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;

/**
 * Created by agibsonccc on 11/30/14.
 */
public class HogWildWorkRouter extends BaseWorkRouter  {
    @Override
    public StateTracker createStateTracker(Configuration conf) throws Exception {
        return new HazelCastStateTracker();
    }

    @Override
    public boolean sendWork() {
        return true;
    }


}
