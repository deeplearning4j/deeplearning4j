package org.deeplearning4j.arbiter.ui.data;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;

/**
 * A {@link org.deeplearning4j.api.storage.Persistable} implemention for model results - i.e., results for
 * each model
 *
 * @author Alex BLack
 */
@Data
public class ModelInfoPersistable extends BaseJavaPersistable {

    private String workerId;
    private Integer modelIdx;
    private Double score;
    private CandidateStatus status;
    private long lastUpdateTime;
    private long numParameters;
    private int numLayers;
    //From candidate generator - this + model hyperparam space means we can work out specific hyperparam
    // settings for this model
    private double[] paramSpaceValues;
    private int totalNumUpdates;
    //Values for score vs. iteration chart
    private int[] iter;
    private float[] scoreVsIter;
    private String modelConfigJson;
    private String exceptionStackTrace;

    public ModelInfoPersistable(String sessionId, String workerId, long timeStamp){
        super(sessionId, timeStamp);

        this.workerId = workerId;
    }

    private ModelInfoPersistable(Builder builder){
        super(builder);
        this.workerId = builder.workerId;
        this.modelIdx = builder.modelIdx;
        this.score = builder.score;
        this.status = builder.status;
        this.iter = builder.iter;
        this.scoreVsIter = builder.scoreVsIter;
        this.lastUpdateTime = builder.lastUpdateTime;
        this.numParameters = builder.numParameters;
        this.numLayers = builder.numLayers;
        this.paramSpaceValues = builder.paramSpaceValues;
        this.modelConfigJson = builder.modelConfigJson;
        this.totalNumUpdates = builder.totalNumUpdates;
        this.exceptionStackTrace = builder.exceptionStackTrace;
    }

    public ModelInfoPersistable(){
        //No-arg costructor for Pesistable encoding/decoding
    }

    @Override
    public String getWorkerID() {
        return workerId;
    }


    public static class Builder extends BaseJavaPersistable.Builder<Builder> {

        private String workerId;
        private Integer modelIdx;
        private Double score;
        private CandidateStatus status;
        private long lastUpdateTime;;
        private long numParameters;
        private int numLayers;
        private int totalNumUpdates;
        private double[] paramSpaceValues;
        private int[] iter;
        private float[] scoreVsIter;
        private String modelConfigJson;
        private String exceptionStackTrace;

        public Builder workerId(String workerId){
            this.workerId = workerId;
            return this;
        }

        public Builder modelIdx(Integer idx){
            this.modelIdx = idx;
            return this;
        }

        public Builder score(Double score){
            this.score = score;
            return this;
        }

        public Builder status(CandidateStatus status){
            this.status = status;
            return this;
        }

        public Builder scoreVsIter(int[] iter, float[] scoreVsIter){
            this.iter = iter;
            this.scoreVsIter = scoreVsIter;
            return this;
        }

        public Builder lastUpdateTime(long lastUpdateTime){
            this.lastUpdateTime = lastUpdateTime;
            return this;
        }

        public Builder numParameters(long numParameters){
            this.numParameters = numParameters;
            return this;
        }

        public Builder numLayers(int numLayers){
            this.numLayers = numLayers;
            return this;
        }

        public Builder totalNumUpdates(int totalNumUpdates){
            this.totalNumUpdates = totalNumUpdates;
            return this;
        }

        public Builder paramSpaceValues(double[] paramSpaceValues){
            this.paramSpaceValues = paramSpaceValues;
            return this;
        }

        public Builder modelConfigJson(String modelConfigJson){
            this.modelConfigJson = modelConfigJson;
            return this;
        }

        public Builder exceptionStackTrace(String exceptionStackTrace){
            this.exceptionStackTrace = exceptionStackTrace;
            return this;
        }

        public ModelInfoPersistable build(){
            return new ModelInfoPersistable(this);
        }
    }
}
