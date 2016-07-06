package io.skymind.echidna.spark.quality.longq;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.io.data.NullWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.dataquality.columns.LongQuality;
import io.skymind.echidna.api.metadata.LongMetaData;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class LongQualityAddFunction implements Function2<LongQuality, Writable, LongQuality> {

    private final LongMetaData meta;

    @Override
    public LongQuality call(LongQuality v1, Writable writable) throws Exception {

        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;
        long nonLong = v1.getCountNonLong();

        if (meta.isValid(writable)) valid++;
        else if (writable instanceof NullWritable || writable instanceof Text && (writable.toString() == null || writable.toString().isEmpty()))
            countMissing++;
        else invalid++;

        String str = writable.toString();
        try {
            Long.parseLong(str);
        } catch (NumberFormatException e) {
            nonLong++;
        }

        return new LongQuality(valid, invalid, countMissing, countTotal, nonLong);
    }
}
