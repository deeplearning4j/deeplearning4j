package io.skymind.echidna.api.sequence;


import lombok.Data;
import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.schema.Schema;
import io.skymind.echidna.api.schema.SequenceSchema;

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
