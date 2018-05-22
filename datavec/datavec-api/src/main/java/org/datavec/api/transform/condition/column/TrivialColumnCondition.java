package org.datavec.api.transform.condition.column;

import lombok.Data;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Created by huitseeker on 5/17/17.
 */
@JsonIgnoreProperties({"schema"})
@Data
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
