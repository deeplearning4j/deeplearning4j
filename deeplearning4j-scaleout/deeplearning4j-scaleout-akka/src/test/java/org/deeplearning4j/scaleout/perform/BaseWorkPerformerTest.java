package org.deeplearning4j.scaleout.perform;

import static org.junit.Assume.*;

import org.deeplearning4j.scaleout.job.Job;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class BaseWorkPerformerTest {


    protected void assumeJobResultNotNull(WorkerPerformer perform,Job j) {
        perform.perform(j);
        assumeNotNull(j.getResult());
    }


}
