package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast;


import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;

import org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.*;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;


/**
 * Tracks state of workers and jobs 
 * via hazelcast distributed data structures
 * @author Adam Gibson
 *
 */

@Path("/statetracker")
@Produces(MediaType.APPLICATION_JSON)
public class HazelCastStateTracker  extends BaseHazelCastStateTracker<UpdateableImpl> {

    public HazelCastStateTracker() throws Exception {
    }

    /**
     * Initializes the state tracker binding to the given port
     *
     * @param stateTrackerPort the port to bind to
     * @throws Exception
     */
    public HazelCastStateTracker(int stateTrackerPort) throws Exception {
        super(stateTrackerPort);
    }

    /**
     * Worker constructor
     *
     * @param connectionString
     */
    public HazelCastStateTracker(String connectionString) throws Exception {
        super(connectionString);
    }

    public HazelCastStateTracker(String connectionString, String type, int stateTrackerPort) throws Exception {
        super(connectionString, type, stateTrackerPort);
    }

    @Override
    public UpdateSaver<UpdateableImpl> createUpdateSaver() {
        return new LocalFileUpdateSaver();
    }

    /**
     * Updates  for mini batches
     *
     * @return the current list of updates for mini batches
     */
    @Override
    public IterateAndUpdate<UpdateableImpl> updates() {
        DeepLearningAccumulator d = new DeepLearningAccumulator(workerUpdates().size());
        DeepLearningAccumulatorIterateAndUpdate d2 = new DeepLearningAccumulatorIterateAndUpdate(d,updateSaver(),workers());
        return d2;
    }

}
