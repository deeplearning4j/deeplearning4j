package org.nd4j.etl4j.spark.transform.filter;

import org.apache.spark.api.java.function.Function;
import org.canova.api.io.data.NullWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;

/**
 * Created by Alex on 6/03/2016.
 */
public class FilterWritablesBySchemaFunction implements Function<Writable,Boolean> {

    private final ColumnMetaData meta;
    private final boolean keepValid;    //If true: keep valid. If false: keep invalid
    private final boolean excludeMissing;   //If true: remove/exclude any


    public FilterWritablesBySchemaFunction(ColumnMetaData meta, boolean keepValid) {
        this(meta,keepValid,false);
    }

    /**
     *
     * @param meta              Column meta data
     * @param keepValid         If true: keep only the valid writables. If false: keep only the invalid writables
     * @param excludeMissing    If true: don't return any missing values, regardless of keepValid setting (i.e., exclude any NullWritable or empty string values)
     */
    public FilterWritablesBySchemaFunction(ColumnMetaData meta, boolean keepValid, boolean excludeMissing) {
        this.meta = meta;
        this.keepValid = keepValid;
        this.excludeMissing = excludeMissing;
    }

    @Override
    public Boolean call(Writable v1) throws Exception {
        boolean valid = meta.isValid(v1);
        if(excludeMissing && (v1 instanceof NullWritable || v1 instanceof Text && (v1.toString() == null || v1.toString().isEmpty()))) return false;    //Remove (spark)
        if(keepValid) return valid; //Spark: return true to keep
        else return !valid;
    }
}
