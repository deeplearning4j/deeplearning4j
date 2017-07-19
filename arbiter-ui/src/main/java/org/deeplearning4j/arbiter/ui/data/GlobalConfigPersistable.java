package org.deeplearning4j.arbiter.ui.data;

import lombok.AllArgsConstructor;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.arbiter.ui.module.ArbiterModule;
import org.deeplearning4j.ui.stats.impl.java.JavaStatsInitializationReport;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;

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
