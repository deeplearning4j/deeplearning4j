package org.nd4j.linalg.util;

import com.google.common.collect.Table;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * Synchronized table
 *
 * @author Adam Gibson
 */
public class SynchronizedTable<R, C, V> implements Table<R, C, V> {
    private Table<R, C, V> wrapped;

    public SynchronizedTable(Table<R, C, V> wrapped) {
        this.wrapped = wrapped;
    }

    @Override
    public synchronized boolean contains(Object rowKey, Object columnKey) {
        return wrapped.contains(rowKey, columnKey);
    }

    @Override
    public synchronized boolean containsRow(Object rowKey) {
        return wrapped.containsRow(rowKey);
    }

    @Override
    public synchronized boolean containsColumn(Object columnKey) {
        return wrapped.containsColumn(columnKey);
    }

    @Override
    public synchronized boolean containsValue(Object value) {
        return wrapped.containsValue(value);
    }

    @Override
    public synchronized V get(Object rowKey, Object columnKey) {
        return get(rowKey, columnKey);
    }

    @Override
    public synchronized boolean isEmpty() {
        return wrapped.isEmpty();
    }

    @Override
    public int size() {
        return wrapped.size();
    }

    @Override
    public synchronized void clear() {
        wrapped.clear();
    }

    @Override
    public synchronized V put(R rowKey, C columnKey, V value) {
        return wrapped.put(rowKey, columnKey, value);
    }

    @Override
    public synchronized void putAll(Table<? extends R, ? extends C, ? extends V> table) {
        wrapped.putAll(table);
    }

    @Override
    public synchronized V remove(Object rowKey, Object columnKey) {
        return wrapped.remove(rowKey, columnKey);
    }

    @Override
    public synchronized Map<C, V> row(R rowKey) {
        return wrapped.row(rowKey);
    }

    @Override
    public synchronized Map<R, V> column(C columnKey) {
        return wrapped.column(columnKey);
    }

    @Override
    public synchronized Set<Cell<R, C, V>> cellSet() {
        return wrapped.cellSet();
    }

    @Override
    public synchronized Set<R> rowKeySet() {
        return wrapped.rowKeySet();
    }

    @Override
    public synchronized Set<C> columnKeySet() {
        return wrapped.columnKeySet();
    }

    @Override
    public synchronized Collection<V> values() {
        return wrapped.values();
    }

    @Override
    public synchronized Map<R, Map<C, V>> rowMap() {
        return wrapped.rowMap();
    }

    @Override
    public synchronized Map<C, Map<R, V>> columnMap() {
        return wrapped.columnMap();
    }
}
