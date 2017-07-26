package org.datavec.api.util.jdbc;

import org.datavec.api.writable.*;

import java.sql.Types;

/**
 * Transform jdbc column data into Writable objects
 *
 * @author Adrien Plagnol
 */
public class JdbcWritableConverter {

    public static Writable convert(final Object columnValue, final int columnType) {
        switch (columnType) {
            case Types.BOOLEAN:
                return new BooleanWritable((boolean)columnValue);

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
            case Types.BIGINT:
            case Types.BIT:
                return new IntWritable((int) columnValue);

            default:
                throw new IllegalArgumentException("Column type unknown");
        }
    }


    private JdbcWritableConverter() {}
}
