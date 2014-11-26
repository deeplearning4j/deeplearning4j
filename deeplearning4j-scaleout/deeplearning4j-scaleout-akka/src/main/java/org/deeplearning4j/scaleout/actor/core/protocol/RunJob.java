package org.deeplearning4j.scaleout.actor.core.protocol;

import java.io.Serializable;


/**
   Alerts a worker a job is available
 * @author Adam Gibson
 */
public class RunJob implements Serializable {

    private static RunJob INSTANCE = new RunJob();

    private RunJob(){}

    public static RunJob getInstance() {
        return INSTANCE;
    }


}
