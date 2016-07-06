package io.skymind.echidna.spark.quality.time;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.io.data.NullWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.dataquality.columns.TimeQuality;
import io.skymind.echidna.api.metadata.TimeMetaData;

@AllArgsConstructor
public class TimeQualityAddFunction implements Function2<TimeQuality, Writable, TimeQuality> {

    private final TimeMetaData meta;

    @Override
    public TimeQuality call(TimeQuality v1, Writable writable) throws Exception {

        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;

        if (meta.isValid(writable)) valid++;
        else if (writable instanceof NullWritable || writable instanceof Text && (writable.toString() == null || writable.toString().isEmpty()))
            countMissing++;
        else invalid++;

        return new TimeQuality(valid, invalid, countMissing, countTotal);
    }
}
