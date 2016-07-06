package io.skymind.echidna.api.transform.column;

import io.skymind.echidna.api.Transform;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.schema.Schema;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Rename one or more columns
 *
 * @author Alex Black
 */
public class RenameColumnsTransform implements Transform {

    private final List<String> oldNames;
    private final List<String> newNames;
    private Schema inputSchema;

    public RenameColumnsTransform(String oldName, String newName){
        this(Collections.singletonList(oldName), Collections.singletonList(newName));
    }

    public RenameColumnsTransform(List<String> oldNames, List<String> newNames ){
        if(oldNames.size() != newNames.size()) throw new IllegalArgumentException("Invalid input: old/new names lists differ in length");
        this.oldNames = oldNames;
        this.newNames = newNames;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<String> inputNames = inputSchema.getColumnNames();
        List<String> outputNames = new ArrayList<>(oldNames.size());

        for(String s : inputNames){
            int idx = oldNames.indexOf(s);
            if(idx >= 0){
                //Switch the old and new names
                outputNames.add(newNames.get(idx));
            } else {
                outputNames.add(s);
            }
        }

        return inputSchema.newSchema(outputNames,inputSchema.getColumnMetaData());
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        //No op
        return writables;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        //No op
        return sequence;
    }

    @Override
    public String toString(){
        return "RenameColumnsTransform(oldNames=" + oldNames + ",newNames=" + newNames + ")";
    }
}
