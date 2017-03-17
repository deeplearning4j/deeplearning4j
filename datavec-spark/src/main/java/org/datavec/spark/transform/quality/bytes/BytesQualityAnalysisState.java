package org.datavec.spark.transform.quality.bytes;

import lombok.Getter;
import org.datavec.api.transform.quality.columns.BytesQuality;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.quality.QualityAnalysisState;

/**
 * Created by huitseeker on 3/6/17.
 * NOTE: this class is not ready for production
 * See the {@link BytesQuality} class.

 */
public class BytesQualityAnalysisState implements QualityAnalysisState<BytesQualityAnalysisState> {

    @Getter
    private BytesQuality bytesQuality;

    public BytesQualityAnalysisState() {
        this.bytesQuality = new BytesQuality();
    }

    public BytesQualityAnalysisState add(Writable writable) throws Exception {
        return this;
    }

    public BytesQualityAnalysisState merge(BytesQualityAnalysisState other) throws Exception {
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return bytesQuality;
    }
}
