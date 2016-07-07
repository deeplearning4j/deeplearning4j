package org.datavec.api.transform.transform.integer;

import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Writable;
import lombok.Data;
import lombok.EqualsAndHashCode;

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
