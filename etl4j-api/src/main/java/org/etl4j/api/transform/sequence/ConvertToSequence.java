package io.skymind.echidna.api.sequence;


import lombok.Data;
import io.skymind.echidna.api.schema.Schema;
import io.skymind.echidna.api.schema.SequenceSchema;

/**
 * Convert a set of values to a sequence
 * Created by Alex on 11/03/2016.
 */
@Data
public class ConvertToSequence {

    private final String keyColumn;
    private final SequenceComparator comparator;    //For sorting values within collected (unsorted) sequence
    private Schema inputSchema;

    public ConvertToSequence(String keyColumn, SequenceComparator comparator){
        this.keyColumn = keyColumn;
        this.comparator = comparator;
    }

    public SequenceSchema transform(Schema schema){
        return new SequenceSchema(schema.getColumnNames(),schema.getColumnMetaData());
    }

    public void setInputSchema(Schema schema){
        this.inputSchema = schema;
        comparator.setSchema(transform(schema));
    }

}
