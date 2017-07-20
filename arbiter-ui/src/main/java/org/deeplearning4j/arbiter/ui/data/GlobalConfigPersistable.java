package org.deeplearning4j.arbiter.ui.data;

import org.deeplearning4j.arbiter.ui.module.ArbiterModule;

/**
 * Created by Alex on 19/07/2017.
 */
public class GlobalConfigPersistable extends BaseJavaPersistable {
    public static final String GLOBAL_WORKER_ID = "global";

    private String sessionId;

    public GlobalConfigPersistable(String sessionId, long  timestamp){
        super(sessionId, timestamp);

    }

    @Override
    public String getTypeID() {
        return ArbiterModule.ARBITER_UI_TYPE_ID;
    }

    @Override
    public String getWorkerID() {
        return GLOBAL_WORKER_ID;
    }
}
