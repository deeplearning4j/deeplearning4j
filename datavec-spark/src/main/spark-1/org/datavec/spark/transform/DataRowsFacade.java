package org.datavec.spark.transform;

import org.apache.spark.sql.DataFrame;

/**
 * Dataframe facade to hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to DataFrame / Dataset
 *
 */
public class DataRowsFacade {

    private final DataFrame df;

    private DataRowsFacade(DataFrame df) {
        this.df = df;
    }

    public static DataRowsFacade dataRows(DataFrame df) {
        return new DataRowsFacade(df);
    }

    public DataFrame get() {
        return df;
    }
}
