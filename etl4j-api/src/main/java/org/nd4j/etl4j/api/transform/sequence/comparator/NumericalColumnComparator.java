package org.nd4j.etl4j.api.transform.sequence.comparator;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.ColumnType;
import org.nd4j.etl4j.api.transform.schema.Schema;

/**
 * Sequence comparator: compare elements in a sequence using the values in a single column
 *
 * Can be applied on any numerical column (Integer, Long, Double or Time columns)
 *
 * @author Alex Black
 */
public class NumericalColumnComparator extends BaseColumnComparator {

    private ColumnType columnType;
    private boolean ascending;

    public NumericalColumnComparator(String columnName){
        this(columnName, true);
    }

    public NumericalColumnComparator(String columnName, boolean ascending){
        super(columnName);
        this.ascending = ascending;
    }

    @Override
    public void setSchema(Schema sequenceSchema){
        super.setSchema(sequenceSchema);
        this.columnType = sequenceSchema.getType(this.columnIdx);
        switch(columnType){
            case Integer:
            case Long:
            case Double:
            case Time:
                //All ok. Time column uses LongWritables too...
                break;
            case Categorical:
            case Bytes:
            case String:
            default:
                throw new IllegalStateException("Cannot apply numerical column comparator on column of type " + columnType);
        }
    }

    @Override
    protected int compare(Writable w1, Writable w2) {
        int compare;
        switch(columnType){
            case Integer:
                compare = Integer.compare(w1.toInt(), w2.toInt());
                break;
            case Time:
            case Long:
                compare = Long.compare(w1.toLong(), w2.toLong());
                break;
            case Double:
                compare = Double.compare(w1.toDouble(), w2.toDouble());
                break;
            default:
                //Should never happen...
                throw new RuntimeException("Cannot apply numerical column comparator on column of type " + columnType);
        }

        if(ascending) return compare;
        return -compare;
    }
}
