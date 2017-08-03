package org.datavec.api.util.jdbc;

import java.sql.Types;
import org.datavec.api.writable.BooleanWritable;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

/**
 * Transform jdbc column data into Writable objects
 *
 * @author Adrien Plagnol
 */
public class JdbcWritableConverter {

    public static Writable convert(final Object columnValue, final int columnType) {
        switch (columnType) {
            case Types.BOOLEAN:
                return new BooleanWritable((boolean) columnValue);

            case Types.DATE:
            case Types.TIME:
            case Types.TIMESTAMP:
            case Types.CHAR:
            case Types.LONGVARCHAR:
            case Types.LONGNVARCHAR:
            case Types.NCHAR:
            case Types.NVARCHAR:
            case Types.VARCHAR:
                return new Text((String) columnValue);

            case Types.DECIMAL:
            case Types.FLOAT:
            case Types.NUMERIC:
            case Types.REAL:
                return new FloatWritable((float) columnValue);

            case Types.DOUBLE:
                return new DoubleWritable((double) columnValue);

            case Types.INTEGER:
            case Types.SMALLINT:
            case Types.TINYINT:
            case Types.BIT:
                return new IntWritable((int) columnValue);

            case Types.BIGINT:
                return new LongWritable((long) columnValue);

            default:
                throw new IllegalArgumentException("Column type unknown");
        }
    }

    private JdbcWritableConverter() {
    }
}
