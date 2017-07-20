package org.deeplearning4j.arbiter.ui.data;

import lombok.Getter;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.ui.misc.JsonMapper;
import org.deeplearning4j.arbiter.ui.module.ArbiterModule;

/**
 * Created by Alex on 19/07/2017.
 */
@Getter
public class GlobalConfigPersistable extends BaseJavaPersistable {
    public static final String GLOBAL_WORKER_ID = "global";

    private String optimizationConfigJson;

    public GlobalConfigPersistable(String sessionId, long  timestamp){
        super(sessionId, timestamp);
    }

    public GlobalConfigPersistable(Builder builder){
        super(builder);
        this.optimizationConfigJson = builder.optimizationConfigJson;
    }

    @Override
    public String getTypeID() {
        return ArbiterModule.ARBITER_UI_TYPE_ID;
    }

    @Override
    public String getWorkerID() {
        return GLOBAL_WORKER_ID;
    }


    public OptimizationConfiguration getOptimizationConfiguration(){
        return JsonMapper.fromJson(optimizationConfigJson, OptimizationConfiguration.class);
    }

    public static class Builder extends BaseJavaPersistable.Builder<Builder>{

        private String optimizationConfigJson;
        private int[] candidateCounts;  //queued, completed, failed, total

        public Builder optimizationConfigJson(String optimizationConfigJson){
            this.optimizationConfigJson = optimizationConfigJson;
            return this;
        }

        public Builder candidateCounts(int queued, int completed, int failed, int total){
            this.candidateCounts = new int[]{queued, completed, failed, total};
            return this;
        }

        public GlobalConfigPersistable build(){
            return new GlobalConfigPersistable(this);
        }

    }
}
