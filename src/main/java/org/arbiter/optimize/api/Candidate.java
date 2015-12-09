package org.arbiter.optimize.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;

/** Candidate: a proposed hyperparameter configuration */
@Data
public class Candidate<T> implements Serializable {

    private T value;

    public Candidate( T value ){
        this.value = value;
    }


}
