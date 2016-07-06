package org.nd4j.etl4j.spark.transform.quality.longq;

import org.apache.spark.api.java.function.Function2;
import org.nd4j.etl4j.api.transform.dataquality.columns.LongQuality;

/**
 * Created by Alex on 5/03/2016.
 */
public class LongQualityMergeFunction implements Function2<LongQuality,LongQuality,LongQuality> {
    @Override
    public LongQuality call(LongQuality v1, LongQuality v2) throws Exception {
        return v1.add(v2);
    }
}
