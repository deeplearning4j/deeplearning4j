package org.datavec.api.transform.transform.doubletransform;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;

/**
 * Convert any value to an Double
 *
 * @author Justin Long (crockpotveggies)
 */
@NoArgsConstructor
@Data
public class ConvertToDouble extends BaseDoubleTransform {

    /**
     * @param column Name of the column to convert to a Double column
     */
    public ConvertToDouble(String column) {
        super(column);
    }

    @Override
    public DoubleWritable map(Writable writable) {
        if(writable.getType() == WritableType.Double){
            return (DoubleWritable)writable;
        }
        return new DoubleWritable(writable.toDouble());
    }


    @Override
    public Object map(Object input) {
        if(input instanceof Number){
            return ((Number) input).doubleValue();
        }
        return Double.parseDouble(input.toString());
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        return new DoubleMetaData(newColumnName);
    }

    @Override
    public String toString() {
        return "ConvertToDouble(columnName=" + columnName + ")";
    }
}
