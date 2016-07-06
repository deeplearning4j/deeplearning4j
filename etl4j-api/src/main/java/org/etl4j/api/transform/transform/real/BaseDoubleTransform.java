package io.skymind.echidna.api.transform.real;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.metadata.DoubleMetaData;
import io.skymind.echidna.api.transform.BaseColumnTransform;

/**
 *
 */
@EqualsAndHashCode(callSuper = true)
@Data
public abstract class BaseDoubleTransform extends BaseColumnTransform {

    public BaseDoubleTransform(String column){
        super(column);
    }

    public abstract Writable map(Writable writable);

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnMeta){
        if(oldColumnMeta instanceof DoubleMetaData) return oldColumnMeta;
        else return new DoubleMetaData();
    }
}
