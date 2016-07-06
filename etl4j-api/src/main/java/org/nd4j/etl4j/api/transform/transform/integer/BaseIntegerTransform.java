package org.nd4j.etl4j.api.transform.transform.integer;

import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnTransform;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.etl4j.api.writable.Writable;

/**
 * Abstract integer transformation (single column)
 */
@EqualsAndHashCode(callSuper = true)
@Data
public abstract class BaseIntegerTransform extends BaseColumnTransform {

    public BaseIntegerTransform(String column){
        super(column);
    }

    public abstract Writable map(Writable writable);

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnMeta){
        return oldColumnMeta;
    }
}
