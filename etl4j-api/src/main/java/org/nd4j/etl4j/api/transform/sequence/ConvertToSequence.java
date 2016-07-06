package org.nd4j.etl4j.api.transform.sequence;


import lombok.Data;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.schema.SequenceSchema;

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
