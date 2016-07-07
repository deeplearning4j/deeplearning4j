package org.nd4j.etl4j.api.transform.transform.condition;

import org.nd4j.etl4j.api.transform.condition.Condition;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.Transform;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.ArrayList;
import java.util.List;

/**
 * Replace the value in a specified column with a new value taken from another column, if a condition is satisfied/true.<br>
 * Note that the condition can be any generic condition, including on other column(s), different to the column
 * that will be modified if the condition is satisfied/true.<br>
 *
 * <b>Note</b>: For sequences, this transform use the convention that each step in the sequence is passed to the condition,
 * and replaced (or not) separately (i.e., Condition.condition(List<Writable>) is used on each time step individually)
 *
 * @author Alex Black
 * @see ConditionalReplaceValueTransform to do a conditional replacement with a fixed value (instead of a value from another column)
 */
public class ConditionalCopyValueTransform implements Transform {

    private final String columnToReplace;
    private final String sourceColumn;
    private final Condition condition;
    private int columnToReplaceIdx = -1;
    private int sourceColumnIdx = -1;

    /**
     *
     * @param columnToReplace    Name of the column in which to replace the old value
     * @param sourceColumn       Name of the column to get the new value from
     * @param condition          Condition
     */
    public ConditionalCopyValueTransform(String columnToReplace, String sourceColumn, Condition condition ){
        this.columnToReplace = columnToReplace;
        this.sourceColumn = sourceColumn;
        this.condition = condition;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //Conditional copy should not change any of the metadata, under normal usage
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        if(!inputSchema.hasColumn(columnToReplace)) throw new IllegalStateException("Column \"" + columnToReplace + "\" not found in input schema");
        if(!inputSchema.hasColumn(sourceColumn)) throw new IllegalStateException("Column \"" + sourceColumn + "\" not found in input schema");
        columnToReplaceIdx = inputSchema.getIndexOfColumn(columnToReplace);
        sourceColumnIdx = inputSchema.getIndexOfColumn(sourceColumn);
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
            newList.set(columnToReplaceIdx,writables.get(sourceColumnIdx));
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
        return "ConditionalReplaceValueTransform(replaceColumn=\"" + columnToReplace + "\",sourceColumn="+ sourceColumn + ",condition=" + condition + ")";
    }
}
