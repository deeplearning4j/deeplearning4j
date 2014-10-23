package org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.deepautoencoder;
import org.deeplearning4j.iterativereduce.akka.DeepAutoEncoderAccumulator;
import org.deeplearning4j.iterativereduce.tracker.statetracker.*;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.*;
import org.deeplearning4j.scaleout.iterativereduce.deepautoencoder.UpdateableEncoderImpl;


/**
 * Tracks state of workers and jobs 
 * via hazelcast distributed data structures
 * @author Adam Gibson
 *
 */

public class DeepAutoEncoderHazelCastStateTracker extends BaseHazelCastStateTracker<UpdateableEncoderImpl> {
    public DeepAutoEncoderHazelCastStateTracker() throws Exception {
    }

    /**
     * Initializes the state tracker binding to the given port
     *
     * @param stateTrackerPort the port to bind to
     * @throws Exception
     */
    public DeepAutoEncoderHazelCastStateTracker(int stateTrackerPort) throws Exception {
        super(stateTrackerPort);
    }

    /**
     * Worker constructor
     *
     * @param connectionString
     */
    public DeepAutoEncoderHazelCastStateTracker(String connectionString) throws Exception {
        super(connectionString);
    }

    public DeepAutoEncoderHazelCastStateTracker(String connectionString, String type, int stateTrackerPort) throws Exception {
        super(connectionString, type, stateTrackerPort);
    }

    @Override
    public UpdateSaver<UpdateableEncoderImpl> createUpdateSaver() {
        return new LocalFileUpdateSaver();
    }

    /**
     * Updates  for mini batches
     *
     * @return the current list of updates for mini batches
     */
    @Override
    public IterateAndUpdate<UpdateableEncoderImpl> updates() {
        DeepAutoEncoderAccumulator d = new DeepAutoEncoderAccumulator(workerUpdates().size());
        DeepAutoEncoderAccumulatorIterateAndUpdate d2 = new DeepAutoEncoderAccumulatorIterateAndUpdate(d,updateSaver(),workers());
        return d2;
    }
}
