package org.nd4j.etl4j.api.transform.transform.condition;

import org.nd4j.etl4j.api.transform.condition.Condition;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.Transform;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.ArrayList;
import java.util.List;

/**
 * Replace the value in a specified column with a new value, if a condition is satisfied/true.<br>
 * Note that the condition can be any generic condition, including on other column(s), different to the column
 * that will be modified if the condition is satisfied/true.<br>
 *
 * <b>Note</b>: For sequences, this transform use the convention that each step in the sequence is passed to the condition,
 * and replaced (or not) separately (i.e., Condition.condition(List<Writable>) is used on each time step individually)
 *
 * @author Alex Black
 * @see ConditionalCopyValueTransform to do a conditional replacement with a value taken from another column
 */
public class ConditionalReplaceValueTransform implements Transform {

    private final String columnToReplace;
    private final Writable newValue;
    private final Condition condition;
    private int columnToReplaceIdx = -1;

    /**
     *
     * @param columnToReplace    Name of the column in which to replace the old value with 'newValue', if the condition holds
     * @param newValue           New value to use
     * @param condition          Condition
     */
    public ConditionalReplaceValueTransform(String columnToReplace, Writable newValue, Condition condition ){
        this.columnToReplace = columnToReplace;
        this.newValue = newValue;
        this.condition = condition;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //Conditional replace should not change any of the metadata, under normal usage
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        columnToReplaceIdx = inputSchema.getColumnNames().indexOf(columnToReplace);
        if(columnToReplaceIdx < 0){
            throw new IllegalStateException("Column \"" + columnToReplace + "\" not found in input schema");
        }
        condition.setInputSchema(inputSchema);
    }

    @Override
    public Schema getInputSchema() {
        return condition.getInputSchema();
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if(condition.condition(writables)){
            //Condition holds -> set new value
            List<Writable> newList = new ArrayList<>(writables);
            newList.set(columnToReplaceIdx,newValue);
            return newList;
        } else {
            //Condition does not hold -> no change
            return writables;
        }
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for(List<Writable> step : sequence){
            out.add(map(step));
        }
        return out;
    }

    @Override
    public String toString(){
        return "ConditionalReplaceValueTransform(replaceColumn=\"" + columnToReplace + "\",newValue="+newValue + ",condition=" + condition + ")";
    }
}
