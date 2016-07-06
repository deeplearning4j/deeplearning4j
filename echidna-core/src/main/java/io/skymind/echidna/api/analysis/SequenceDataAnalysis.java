package io.skymind.echidna.api.analysis;

import io.skymind.echidna.api.analysis.columns.ColumnAnalysis;
import lombok.Data;
import lombok.EqualsAndHashCode;
import io.skymind.echidna.api.analysis.sequence.SequenceLengthAnalysis;
import io.skymind.echidna.api.schema.Schema;

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
