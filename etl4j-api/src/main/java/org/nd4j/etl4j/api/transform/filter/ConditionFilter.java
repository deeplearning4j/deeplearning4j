package org.nd4j.etl4j.api.transform.filter;

import org.nd4j.etl4j.api.transform.condition.Condition;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.List;

/**
 * A filter based on a {@link Condition}.<br>
 * If condition is satisfied (returns true): remove the example or sequence<br>
 * If condition is not satisfied (returns false): keep the example or sequence
 *
 * @author Alex Black
 */
public class ConditionFilter implements Filter {

    private final Condition condition;

    public ConditionFilter(Condition condition){
        this.condition = condition;
    }

    @Override
    public boolean removeExample(List<Writable> writables) {
        return condition.condition(writables);
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        return condition.conditionSequence(sequence);
    }

    @Override
    public void setInputSchema(Schema schema) {
        condition.setInputSchema(schema);
    }

    @Override
    public Schema getInputSchema() {
        return condition.getInputSchema();
    }
}
