package io.skymind.echidna.spark.quality.categorical;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.io.data.NullWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.dataquality.columns.CategoricalQuality;
import io.skymind.echidna.api.metadata.CategoricalMetaData;

/**
 * Created by Alex on 5/03/2016.
 */
@AllArgsConstructor
public class CategoricalQualityAddFunction implements Function2<CategoricalQuality,Writable,CategoricalQuality> {

    private final CategoricalMetaData meta;

    @Override
    public CategoricalQuality call(CategoricalQuality v1, Writable writable) throws Exception {

        long valid = v1.getCountValid();
        long invalid = v1.getCountInvalid();
        long countMissing = v1.getCountMissing();
        long countTotal = v1.getCountTotal() + 1;

        if(meta.isValid(writable)) valid++;
        else if(writable instanceof NullWritable || writable instanceof Text && (writable.toString() == null || writable.toString().isEmpty())) countMissing++;
        else invalid++;

        return new CategoricalQuality(valid,invalid,countMissing,countTotal);
    }
}
