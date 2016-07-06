package io.skymind.echidna.api.transform.string;

import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.metadata.StringMetaData;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.transform.BaseColumnTransform;


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
