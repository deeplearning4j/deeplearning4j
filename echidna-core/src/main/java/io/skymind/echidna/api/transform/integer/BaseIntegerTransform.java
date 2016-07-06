package io.skymind.echidna.api.transform.integer;

import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.transform.BaseColumnTransform;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.writable.Writable;

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
