package org.arbiter.optimize.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

@AllArgsConstructor @Data
public class OptimizationResult<T,M> implements Serializable {

    private T config;
    private M result;
    private double score;


}
