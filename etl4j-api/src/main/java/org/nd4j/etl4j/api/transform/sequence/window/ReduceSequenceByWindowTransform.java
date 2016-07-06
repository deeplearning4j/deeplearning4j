package org.nd4j.etl4j.api.transform.sequence.window;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.Transform;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.reduce.IReducer;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.schema.SequenceSchema;

import java.util.ArrayList;
import java.util.List;

/**
 * Idea: do two things.
 * First, apply a window function to the sequence data.
 * Second: Reduce that window of data into a single value by using a Reduce function
 *
 * @author Alex Black
 */
public class ReduceSequenceByWindowTransform implements Transform {

    private IReducer reducer;
    private WindowFunction windowFunction;
    private Schema inputSchema;

    public ReduceSequenceByWindowTransform(IReducer reducer, WindowFunction windowFunction){
        this.reducer = reducer;
        this.windowFunction = windowFunction;
    }


    @Override
    public Schema transform(Schema inputSchema) {
        if(inputSchema != null && !(inputSchema instanceof SequenceSchema)){
            throw new IllegalArgumentException("Invalid input: input schema must be a SequenceSchema");
        }

        //Some window functions may make changes to the schema (adding window start/end times, for example)
        inputSchema = windowFunction.transform(inputSchema);

        //Approach here: The reducer gives us a schema for one time step -> simply convert this to a sequence schema...
        Schema oneStepSchema = reducer.transform(inputSchema);
        List<String> columnNames = oneStepSchema.getColumnNames();
        List<ColumnMetaData> meta = oneStepSchema.getColumnMetaData();

        return new SequenceSchema(columnNames,meta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
        this.windowFunction.setInputSchema(inputSchema);
        reducer.setInputSchema(windowFunction.transform(inputSchema));
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        throw new UnsupportedOperationException("ReduceSequenceByWindownTransform can only be applied on sequences");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {

        //List of windows, which are all small sequences...
        List<List<List<Writable>>> sequenceAsWindows = windowFunction.applyToSequence(sequence);

        List<List<Writable>> out = new ArrayList<>();

        for(List<List<Writable>> window : sequenceAsWindows ){
            List<Writable> reduced = reducer.reduce(window);
            out.add(reduced);
        }

        return out;
    }
}
