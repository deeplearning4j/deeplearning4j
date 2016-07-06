package org.nd4j.etl4j.api.transform.dataquality.columns;

/**
 * Quality of a Bytes column
 *
 * @author Alex Black
 */
public class BytesQuality extends ColumnQuality {

    public BytesQuality() {
        this(0, 0, 0, 0);
    }

    public BytesQuality(long countValid, long countInvalid, long countMissing, long countTotal) {
        super(countValid, countInvalid, countMissing, countTotal);
    }

    @Override
    public String toString() {
        return "BytesQuality(" + super.toString() + ")";
    }

}
