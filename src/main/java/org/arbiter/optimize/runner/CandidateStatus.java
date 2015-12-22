package org.arbiter.optimize.runner;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor @Data
public class CandidateStatus {

    public enum Status {Created,Running,Complete,Failed,Cancelled};

    public CandidateStatus(){
        //No arg constructor for Jackson
    }

    private int index;
    private Status status;
    private Double score;
    private long createdTime;
    private Long startTime;
    private Long endTime;



}
