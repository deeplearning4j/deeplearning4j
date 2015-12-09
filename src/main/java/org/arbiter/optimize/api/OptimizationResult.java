package org.arbiter.optimize.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

@Data
public class OptimizationResult<T,M> implements Serializable {

    private T config;
    private M result;
    private double score;

    public OptimizationResult(T config, M result, double score){
        this.config = config;
        this.result = result;
        this.score = score;
    }


}
