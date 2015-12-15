package org.arbiter.optimize.api;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;
import java.util.Map;

/**Candidate: a proposed hyperparameter configuration.
 * Also includes a map for data parameters, to configure things like data preprocessing, etc.
 *  */
@Data
public class Candidate<T> implements Serializable {

    private T value;
    private int index;
    private Map<String,Object> dataParameters;

    public Candidate( T value, int index ) {
        this(value, index, null);
    }

    public Candidate(T value, int index, Map<String,Object> dataParameters){
        this.value = value;
        this.index = index;
        this.dataParameters = dataParameters;
    }


}
