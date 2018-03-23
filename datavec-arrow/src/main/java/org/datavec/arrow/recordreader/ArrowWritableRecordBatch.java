package org.datavec.arrow.recordreader;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.AbstractWritableRecordBatch;
import org.datavec.arrow.ArrowConverter;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;

/**
 *
 */
@Data
@AllArgsConstructor
public class ArrowWritableRecordBatch extends AbstractWritableRecordBatch implements List<List<Writable>>,Closeable {

    private List<FieldVector> list;
    private int size;
    private Schema schema;
    private ArrowRecordBatch arrowRecordBatch;
    private VectorSchemaRoot vectorLoader;
    private VectorUnloader unloader;

    /**
     * An index in to an individual
     * {@link ArrowRecordBatch}
     * @param list the list of field vectors to use
     * @param schema the schema to use
     */
    public ArrowWritableRecordBatch(List<FieldVector> list, Schema schema) {
        this.list = list;
        this.schema = schema;
        //each column should have same number of rows
        this.size = list.get(0).getValueCount();
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public List<Writable> get(int i) {
        List<Writable> ret = new ArrayList<>(schema.numColumns());
        for(int column = 0; column < schema.numColumns(); column++) {
            ret.add(ArrowConverter.fromEntry(i,list.get(column),schema.getType(column)));
        }
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        ArrowWritableRecordBatch lists = (ArrowWritableRecordBatch) o;
        return size == lists.size &&
                Objects.equals(list, lists.list) &&
                Objects.equals(schema, lists.schema);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), list, size, schema);
    }

    @Override
    public void close() throws IOException {
        if(arrowRecordBatch != null)
            arrowRecordBatch.close();
        if(vectorLoader != null)
            vectorLoader.close();
    }

}
