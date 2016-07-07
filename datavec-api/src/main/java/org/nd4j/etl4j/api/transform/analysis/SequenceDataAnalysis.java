package org.nd4j.etl4j.api.transform.analysis;

import org.nd4j.etl4j.api.transform.analysis.columns.ColumnAnalysis;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.etl4j.api.transform.analysis.sequence.SequenceLengthAnalysis;
import org.nd4j.etl4j.api.transform.schema.Schema;

import java.util.List;

/**
 * Created by Alex on 12/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class SequenceDataAnalysis extends DataAnalysis {

    private final SequenceLengthAnalysis sequenceLengthAnalysis;

    public SequenceDataAnalysis(Schema schema, List<ColumnAnalysis> columnAnalysis, SequenceLengthAnalysis sequenceAnalysis) {
        super(schema, columnAnalysis);
        this.sequenceLengthAnalysis = sequenceAnalysis;
    }

    @Override
    public String toString(){
        return sequenceLengthAnalysis + "\n" + super.toString();
    }
}
