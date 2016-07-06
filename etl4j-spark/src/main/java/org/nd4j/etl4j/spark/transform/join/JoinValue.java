package org.nd4j.etl4j.spark.transform.join;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.canova.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**
 * Simple helper class for executing joins
 */
@AllArgsConstructor @Data
public class JoinValue implements Serializable {

    private final boolean left;
    private final List<Writable> values;

}
