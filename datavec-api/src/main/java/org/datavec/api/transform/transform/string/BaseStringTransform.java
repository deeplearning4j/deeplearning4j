package org.datavec.api.transform.transform.string;

import org.datavec.api.io.data.Text;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.transform.BaseColumnTransform;


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
