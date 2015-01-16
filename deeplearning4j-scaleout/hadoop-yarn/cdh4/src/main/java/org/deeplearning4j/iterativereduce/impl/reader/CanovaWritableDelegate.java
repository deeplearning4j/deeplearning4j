package org.deeplearning4j.iterativereduce.impl.reader;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 *  Canova writable: delegates to the underlying canova writable
 *
 *  @author Adam Gibson
 */
public class CanovaWritableDelegate implements Writable {
    private org.canova.api.writable.Writable writable;

    public CanovaWritableDelegate(org.canova.api.writable.Writable writable) {
        this.writable = writable;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        writable.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {

    }

    @Override
    public String toString() {
        return writable.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CanovaWritableDelegate)) return false;

        CanovaWritableDelegate that = (CanovaWritableDelegate) o;

        if (writable != null ? !writable.equals(that.writable) : that.writable != null) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return writable != null ? writable.hashCode() : 0;
    }
}
