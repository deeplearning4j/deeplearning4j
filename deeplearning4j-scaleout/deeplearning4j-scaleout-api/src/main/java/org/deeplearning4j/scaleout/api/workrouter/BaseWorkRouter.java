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

package org.deeplearning4j.scaleout.api.workrouter;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.api.statetracker.IterateAndUpdate;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.job.Job;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles job flow
 * @author Adam Gibson
 */
public abstract class BaseWorkRouter implements WorkRouter {

    protected StateTracker stateTracker;
    protected boolean waitForWorkers = true;
    protected static final Logger log = LoggerFactory.getLogger(WorkRouter.class);

    public BaseWorkRouter() {

    }

    public BaseWorkRouter(StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }

    @Override
    public void update() {
        Job masterResults = compute();
        log.info("Updating next batch");
        try {
            stateTracker.setCurrent(masterResults);
        } catch (Exception e) {
            e.printStackTrace();
        }

        for(String s : stateTracker.workers()) {
            log.info("Replicating new work to " + s);
            stateTracker.addReplicate(s);
            stateTracker.enableWorker(s);

        }
        stateTracker.workerUpdates().clear();


    }


    public  Job compute() {


        IterateAndUpdate update =  stateTracker.updates();
        if(stateTracker.workerUpdates().isEmpty())
            return null;

        try {
            update.accumulate();

        }catch(Exception e) {
            log.debug("Unable to accumulate results",e);
            return null;
        }

        Job masterResults = null;
        try {
            masterResults = (Job) stateTracker.getCurrent();
        } catch (Exception e) {
            e.printStackTrace();
        }
        if(masterResults == null)
            masterResults = update.accumulated();


        try {
            stateTracker.setCurrent(masterResults);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


        return masterResults;
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

    @Override
    public void routeJob(Job job) {
        try {
            stateTracker.addJobToCurrent(job);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public abstract StateTracker createStateTracker(Configuration conf) throws Exception;


}
