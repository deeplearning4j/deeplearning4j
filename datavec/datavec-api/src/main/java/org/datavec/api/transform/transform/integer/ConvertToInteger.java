package org.datavec.api.transform.transform.integer;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;

/**
 * Convert any value to an Integer.
 *
 * @author Justin Long (crockpotveggies)
 */
@Data
@NoArgsConstructor
public class ConvertToInteger extends BaseIntegerTransform {

    /**
     *
     * @param column Name of the column to convert to an integer
     */
    public ConvertToInteger(String column) {
        super(column);
    }

    @Override
    public IntWritable map(Writable writable) {
        if(writable.getType() == WritableType.Int){
            return (IntWritable)writable;
        }
        return new IntWritable(writable.toInt());
    }

    @Override
    public Object map(Object input) {
        if(input instanceof Number){
            return ((Number) input).intValue();
        }
        return Integer.parseInt(input.toString());
    }


    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        return new IntegerMetaData(newColumnName);
    }

    @Override
    public String toString() {
        return "ConvertToInteger(columnName=" + columnName + ")";
    }
}
