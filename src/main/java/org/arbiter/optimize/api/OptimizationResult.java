package org.arbiter.optimize.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

@Data
public class OptimizationResult<C,M,A> implements Serializable {

    private Candidate<C> candidate;
    private M result;
    private double score;
    private int index;
    private A modelSpecificResults;

    public OptimizationResult(Candidate<C> candidate, M result, double score, int index, A modelSpecificResults){
        this.candidate = candidate;
        this.result = result;
        this.score = score;
        this.index = index;
        this.modelSpecificResults = modelSpecificResults;
    }


}
