package org.datavec.arrow.recordreader;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.AbstractTimeSeriesWritableRecordBatch;
import org.datavec.arrow.ArrowConverter;
import org.jetbrains.annotations.NotNull;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;

/**
 *
 */
@Data
@AllArgsConstructor
public class ArrowWritableRecordTimeSeriesBatch extends AbstractTimeSeriesWritableRecordBatch implements Closeable {

    private List<FieldVector> list;
    private int size;
    private Schema schema;
    private ArrowRecordBatch arrowRecordBatch;
    private VectorSchemaRoot vectorLoader;
    private VectorUnloader unloader;
    private int timeSeriesStride;

    /**
     * An index in to an individual
     * {@link ArrowRecordBatch}
     * @param list the list of field vectors to use
     * @param schema the schema to use
     */
    public ArrowWritableRecordTimeSeriesBatch(List<FieldVector> list, Schema schema,int timeSeriesStride) {
        this.list = list;
        this.schema = schema;
        //each column should have same number of rows
        this.timeSeriesStride = timeSeriesStride;
        this.size = list.size() * list.get(0).getValueCount() / timeSeriesStride;

    }

    public List<List<List<Writable>>> toArrayList() {
        List<List<List<Writable>>> ret = new ArrayList<>();
        for(int i = 0; i < size(); i++) {
            List<List<Writable>> timeStep = get(i);
            List<List<Writable>> addTimeStep = new ArrayList<>();
            for(int j = 0 ; j < timeStep.size(); j++) {
                List<Writable> addingFrom = timeStep.get(j);
                List<Writable> currRecord = new ArrayList<>(addingFrom);
                addTimeStep.add(currRecord);
            }

            ret.add(addTimeStep);
        }

        return ret;
    }


    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public boolean contains(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Iterator<List<List<Writable>>> iterator() {
        return new ArrowListIterator();
    }

    @Override
    public Object[] toArray() {
        Object[] ret = new Object[size()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = get(i);
        }
        return ret;
    }

    @Override
    public <T> T[] toArray(T[] ts) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(List<List<Writable>> writable) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        return false;
    }

    @Override
    public boolean addAll(Collection<? extends List<List<Writable>>> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(int i,  Collection<? extends List<List<Writable>>> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {

    }

    @Override
    public List<List<Writable>> get(int i) {
        ArrowWritableRecordBatch ret =  new ArrowWritableRecordBatch(list,schema,i ,timeSeriesStride / schema.numColumns());
        return ret;
    }

    @Override
    public List<List<Writable>> set(int i, List<List<Writable>> writable) {
        ArrowWritableRecordBatch arrowWritableRecordBatch = (ArrowWritableRecordBatch) get(i);
        for(int batch = 0; batch < writable.size(); batch++) {
            arrowWritableRecordBatch.set(batch,writable.get(i));
        }

        return arrowWritableRecordBatch;
    }

    @Override
    public void add(int i, List<List<Writable>> writable) {
        throw new UnsupportedOperationException();

    }

    @Override
    public List<List<Writable>> remove(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int indexOf(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int lastIndexOf(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ListIterator<List<List<Writable>>> listIterator() {
        return new ArrowListIterator();
    }

    @Override
    public ListIterator<List<List<Writable>>> listIterator(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<List<List<Writable>>> subList(int i, int i1) {
        return null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        ArrowWritableRecordTimeSeriesBatch lists = (ArrowWritableRecordTimeSeriesBatch) o;
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


    private class ArrowListIterator implements ListIterator<List<List<Writable>>> {
        private int index;

        @Override
        public boolean hasNext() {
            return index < size;
        }

        @Override
        public List<List<Writable>> next() {
            return get(index++);
        }

        @Override
        public boolean hasPrevious() {
            return index > 0;
        }

        @Override
        public List<List<Writable>> previous() {
            return get(index - 1);
        }

        @Override
        public int nextIndex() {
            return index + 1;
        }

        @Override
        public int previousIndex() {
            return index - 1;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void set(List<List<Writable>> writables) {
            ArrowWritableRecordTimeSeriesBatch.this.set(index,writables);
        }

        @Override
        public void add(List<List<Writable>> writables) {
            throw new UnsupportedOperationException();

        }
    }


}
