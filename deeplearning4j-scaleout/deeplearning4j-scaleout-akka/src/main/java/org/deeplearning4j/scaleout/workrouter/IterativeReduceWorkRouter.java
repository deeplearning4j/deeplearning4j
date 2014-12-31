package org.deeplearning4j.scaleout.workrouter;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.api.workrouter.BaseWorkRouter;
import org.deeplearning4j.nn.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;

import java.util.Collection;
import java.util.List;

/**
 * Wait for workers being done when routing work
 * @author Adam Gibson
 */
public class IterativeReduceWorkRouter extends BaseWorkRouter {
    public IterativeReduceWorkRouter() {
    }

    public IterativeReduceWorkRouter(StateTracker stateTracker) {
        super(stateTracker);
    }

    @Override
    public StateTracker createStateTracker(Configuration conf) throws Exception {
        return new HazelCastStateTracker();
    }

    @Override
    public boolean sendWork() {
        try {
            List<Job> currentJobs = stateTracker.currentJobs();
            Collection<String> updates = stateTracker.workerUpdates();
            return updates.size() >= stateTracker.workers().size() ||
                    currentJobs.isEmpty() || !isWaitForWorkers();
        }catch(Exception e) {

        }
        return false;
    }


}
