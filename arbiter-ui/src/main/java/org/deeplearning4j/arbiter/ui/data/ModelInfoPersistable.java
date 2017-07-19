package org.deeplearning4j.arbiter.ui.data;

/**
 * Created by Alex on 19/07/2017.
 */
public class ModelInfoPersistable extends BaseJavaPersistable {

    private String workerId;

    public ModelInfoPersistable(String sessionId, String workerId, long timeStamp){
        super(sessionId, timeStamp);

        this.workerId = workerId;
    }

    @Override
    public String getWorkerID() {
        return workerId;
    }
}
