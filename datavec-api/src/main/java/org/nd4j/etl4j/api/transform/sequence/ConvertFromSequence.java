package org.nd4j.etl4j.api.transform.sequence;


import lombok.Data;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.schema.Schema;
import org.nd4j.etl4j.api.transform.schema.SequenceSchema;

import java.util.ArrayList;
import java.util.List;

/**
 * Split up the values in sequences to a set of individual values.<br>
 * i.e., sequences are split up, such that each time step in the sequence is treated as a separate example
 *
 * @author Alex Black
 */
@Data
public class ConvertFromSequence {

    private SequenceSchema inputSchema;

    public ConvertFromSequence(){

    }

    public Schema transform(SequenceSchema schema){

        List<String> names = new ArrayList<>(schema.getColumnNames());
        List<ColumnMetaData> meta = new ArrayList<>(schema.getColumnMetaData());

        return new Schema(names,meta);
    }



}
