package org.datavec.api.transform.transform.real;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.transform.BaseColumnTransform;

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
