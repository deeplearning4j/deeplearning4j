package org.arbiter.optimize.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

@Data
public class OptimizationResult<T,M> implements Serializable {

    private Candidate<T> candidate;
    private M result;
    private double score;
    private int index;

    public OptimizationResult(Candidate<T> candidate, M result, double score, int index){
        this.candidate = candidate;
        this.result = result;
        this.score = score;
        this.index = index;
    }


}
