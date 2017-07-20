package org.deeplearning4j.arbiter.ui.data;

import lombok.Data;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;

/**
 * Created by Alex on 19/07/2017.
 */
@Data
public class ModelInfoPersistable extends BaseJavaPersistable {

    private String workerId;
    private Integer modelIdx;
    private Double score;
    private CandidateStatus status;


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
    }

    @Override
    public String getWorkerID() {
        return workerId;
    }


    public static class Builder extends BaseJavaPersistable.Builder<Builder> {

        protected String workerId;
        protected Integer modelIdx;
        protected Double score;
        protected CandidateStatus status;

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

        public ModelInfoPersistable build(){
            return new ModelInfoPersistable(this);
        }
    }
}
