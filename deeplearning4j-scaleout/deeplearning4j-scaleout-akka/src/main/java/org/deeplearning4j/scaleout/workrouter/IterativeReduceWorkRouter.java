/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.workrouter;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.api.workrouter.BaseWorkRouter;
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
            Collection<String> workers = stateTracker.workers();
            return updates.size() >= workers.size() ||
                    currentJobs.isEmpty() || !isWaitForWorkers();
        }catch(Exception e) {

        }
        return false;
    }


}
