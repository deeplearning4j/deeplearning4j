package org.nd4j.etl4j.api.transform.transform.real;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.DoubleMetaData;
import org.nd4j.etl4j.api.transform.transform.BaseColumnTransform;

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
