package org.nd4j.etl4j.api.transform.transform.string;

import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import org.nd4j.etl4j.api.transform.metadata.StringMetaData;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.transform.BaseColumnTransform;


/**
 * Abstract String column transform
 */
@EqualsAndHashCode(callSuper = true)
@Data
public abstract class BaseStringTransform extends BaseColumnTransform {

    public BaseStringTransform(String column){
        super(column);
    }

    public abstract Text map(Writable writable);

    @Override
    public ColumnMetaData getNewColumnMetaData(ColumnMetaData oldColumnType){
        return new StringMetaData();
    }
}
