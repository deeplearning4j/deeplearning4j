package org.datavec.spark.transform.quality.time;

import org.apache.spark.api.java.function.Function2;
import org.datavec.api.transform.dataquality.columns.TimeQuality;

/**
 * Created by Alex on 5/03/2016.
 */
public class TimeQualityMergeFunction implements Function2<TimeQuality,TimeQuality,TimeQuality> {
    @Override
    public TimeQuality call(TimeQuality v1, TimeQuality v2) throws Exception {
        return v1.add(v2);
    }
}
