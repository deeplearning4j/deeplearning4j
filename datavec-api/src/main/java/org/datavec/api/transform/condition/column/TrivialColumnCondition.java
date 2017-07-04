package org.datavec.api.transform.condition.column;

import org.datavec.api.transform.condition.column.BaseColumnCondition;
import org.datavec.api.transform.condition.column.ColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Created by huitseeker on 5/17/17.
 */
@JsonIgnoreProperties({"schema"})
public class TrivialColumnCondition extends BaseColumnCondition {

    private Schema schema;

    public TrivialColumnCondition(@JsonProperty("name") String name) {
        super(name, DEFAULT_SEQUENCE_CONDITION_MODE);
    }

    @Override
    public String toString() {
        return "Trivial(" + super.columnName + ")";
    }

    @Override
    public boolean columnCondition(Writable writable) {
        return true;
    }

    @Override
    public boolean condition(List<Writable> writables) {
        return true;
    }

    @Override
    public boolean condition(Object input) {
        return true;
    }
}
