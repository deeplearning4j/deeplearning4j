package org.nd4j.etl4j.spark.transform.join;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.etl4j.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**
 * Simple helper class for executing joins
 */
@AllArgsConstructor @Data
public class JoinedValue implements Serializable {

    private final boolean haveLeft;
    private final boolean haveRight;
    private final List<Writable> values;

}
