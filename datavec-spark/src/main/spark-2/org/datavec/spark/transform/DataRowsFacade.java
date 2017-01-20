package org.datavec.spark.transform;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * Dataframe facade to hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to DataFrame / Dataset
 *
 */
public class DataRowsFacade {

    private final Dataset<Row> df;

    private DataRowsFacade(Dataset<Row> df) {
        this.df = df;
    }

    public static DataRowsFacade dataRows(Dataset<Row> df) {
        return new DataRowsFacade(df);
    }

    public Dataset<Row> get() {
        return df;
    }
}
