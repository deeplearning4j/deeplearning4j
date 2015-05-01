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

package org.deeplearning4j.scaleout.statetracker.hazelcast;


import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.api.statetracker.IterateAndUpdate;
import org.deeplearning4j.scaleout.api.statetracker.UpdateSaver;
import org.deeplearning4j.scaleout.statetracker.updatesaver.LocalFileUpdateSaver;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;


/**
 * Tracks state of workers and jobs 
 * via hazelcast distributed data structures
 * @author Adam Gibson
 *
 */

@Path("/statetracker")
@Produces(MediaType.APPLICATION_JSON)
public class HazelCastStateTracker  extends BaseHazelCastStateTracker {

    public HazelCastStateTracker() throws Exception {
        super(DEFAULT_HAZELCAST_PORT);
        setUpdateSaver(createUpdateSaver());
    }

    /**
     * Initializes the state tracker binding to the given port
     *
     * @param stateTrackerPort the port to bind to
     * @throws Exception
     */
    public HazelCastStateTracker(int stateTrackerPort) throws Exception {
        super(stateTrackerPort);
        setUpdateSaver(createUpdateSaver());

    }

    /**
     * Worker constructor
     *
     * @param connectionString
     */
    public HazelCastStateTracker(String connectionString) throws Exception {
        super(connectionString);
        setUpdateSaver(createUpdateSaver());
    }

    public HazelCastStateTracker(String connectionString, String type, int stateTrackerPort) throws Exception {
        super(connectionString, type, stateTrackerPort);
        setUpdateSaver(createUpdateSaver());

    }

    @Override
    public UpdateSaver createUpdateSaver() {
        return new LocalFileUpdateSaver(".",getH());
    }


    @Override
    public Job loadForWorker(String workerId) {
        return null;
    }

    @Override
    public void saveWorker(String workerId, Job d) {

    }

    /**
     * Updates  for mini batches
     *
     * @return the current list of updates for mini batches
     */
    @Override
    public IterateAndUpdate updates() {
        IterateAndUpdateImpl d2 = new IterateAndUpdateImpl(jobAggregator,updateSaver(),workers());
        return d2;
    }

}
