package org.datavec.arrow.recordreader;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.arrow.ArrowConverter;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;

/**
 *
 */
@Data
@AllArgsConstructor
public class ArrowWritableRecordBatch implements List<List<Writable>>,Closeable {

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
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public boolean contains(Object o) {
        return false;
    }

    @Override
    public Iterator<List<Writable>> iterator() {
        return new ArrowListIterator();
    }

    @Override
    public Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    @Override
    public <T> T[] toArray(T[] ts) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(List<Writable> writable) {
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
    public boolean addAll(Collection<? extends List<Writable>> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(int i,  Collection<? extends List<Writable>> collection) {
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
    public List<Writable> get(int i) {
        List<Writable> ret = new ArrayList<>(schema.numColumns());
        for(int column = 0; column < schema.numColumns(); column++) {
            ret.add(ArrowConverter.fromEntry(i,list.get(column),schema.getType(column)));
        }
        return ret;
    }

    @Override
    public List<Writable> set(int i, List<Writable> writable) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void add(int i, List<Writable> writable) {
        throw new UnsupportedOperationException();

    }

    @Override
    public List<Writable> remove(int i) {
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
    public ListIterator<List<Writable>> listIterator() {
        return new ArrowListIterator();
    }

    @Override
    public ListIterator<List<Writable>> listIterator(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<List<Writable>> subList(int i, int i1) {
        throw new UnsupportedOperationException();
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


    private class ArrowListIterator implements ListIterator<List<Writable>> {
        private int index;

        @Override
        public boolean hasNext() {
            return index < size;
        }

        @Override
        public List<Writable> next() {
            return get(index++);
        }

        @Override
        public boolean hasPrevious() {
            return index > 0;
        }

        @Override
        public List<Writable> previous() {
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
        public void set(List<Writable> writables) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void add(List<Writable> writables) {
            throw new UnsupportedOperationException();

        }
    }

}
