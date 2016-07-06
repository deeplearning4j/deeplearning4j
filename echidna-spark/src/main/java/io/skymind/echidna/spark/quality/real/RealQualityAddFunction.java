package io.skymind.echidna.spark.quality.real;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.io.data.NullWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.dataquality.columns.DoubleQuality;
import io.skymind.echidna.api.metadata.DoubleMetaData;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class RealQualityAddFunction implements Function2<DoubleQuality, Writable, DoubleQuality> {

    private final DoubleMetaData meta;

    @Override
    public DoubleQuality call(DoubleQuality v1, Writable writable) throws Exception {

        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;
        long nonReal = v1.getCountNonReal();
        long nan = v1.getCountNaN();
        long infinite = v1.getCountInfinite();

        if (meta.isValid(writable)) valid++;
        else if (writable instanceof NullWritable || writable instanceof Text && (writable.toString() == null || writable.toString().isEmpty()))
            countMissing++;
        else invalid++;

        String str = writable.toString();
        double d;
        try {
            d = Double.parseDouble(str);
            if (Double.isNaN(d)) nan++;
            if (Double.isInfinite(d)) infinite++;
        } catch (NumberFormatException e) {
            nonReal++;
        }

        return new DoubleQuality(valid, invalid, countMissing, countTotal, nonReal, nan, infinite);
    }
}
