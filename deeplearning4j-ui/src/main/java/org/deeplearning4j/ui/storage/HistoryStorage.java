package org.deeplearning4j.ui.storage;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.berkeley.Pair;

import java.util.*;

/**
 * Simple abstract in-memory storage with history option to be used across threads using UiServer.
 *
 * PLEASE NOTE: Storage has no idea what's stored in it's value field, and proper type cast use is assumed.
 *
 * @author raver119@gmail.com
 */
public class HistoryStorage {
    public enum ObjectType {
        TSNE,
        SCORES,
        ACTIVATIONS
    }

    public enum TargetVersion {
        LATEST,
        ANY,
        OLDEST
    }

    public enum SortOutput {
        DESCENDING,
        ASCENDING,
        NONE
    }
    /*
            TODO: it's probably worth making key object Enum too, to avoid misuse and misunderstanding here. To be investigated
    */
    // simple storage here: Key, Version (Major.Minor), Object
    private Table<Object, Pair<Integer, Integer>, Object> historyTable = HashBasedTable.create();

    private static HistoryStorage ourInstance = new HistoryStorage();

    public static HistoryStorage getInstance() {
        return ourInstance;
    }

    private HistoryStorage() {
    }

    public Object get(Object key, TargetVersion version) {
            Map<Pair<Integer, Integer>, Object>  map = historyTable.row(key);
            if (map.size() == 1) {
                // if map has only one value, we'll go straight for it
                return map.values().iterator().next();
            } else if (map.size() > 0) {
                List<Object> objects = getSorted(key, SortOutput.DESCENDING);
                if (version.equals(TargetVersion.OLDEST)) return objects.get(objects.size() - 1);
                    else if (version.equals(TargetVersion.LATEST)) return objects.get(0);
            }
        return null;
    }

    /**
     * This method returns all elements stored with specified key, with some kind of sort being applied
     *
     * @param key
     * @param sortOutput
     * @return
     */
    public List<Object> getSorted(@NonNull Object key, SortOutput sortOutput) {
        List<Object> results = new ArrayList<>();

        switch (sortOutput) {
            case ASCENDING: {
                Map<Pair<Integer, Integer>, Object> map = historyTable.row(key);
                List<SortableObject> list = new ArrayList<>();
                for (Map.Entry<Pair<Integer, Integer>, Object> entry : map.entrySet()) {
                    list.add(new SortableObject(entry.getKey(), entry.getValue()));
                }
                Collections.sort(list, new AscendingComparator());
                results = stripVersions(list);
                break;
            }
            case DESCENDING: {
                Map<Pair<Integer, Integer>, Object> map = historyTable.row(key);
                List<SortableObject> list = new ArrayList<>();
                for (Map.Entry<Pair<Integer, Integer>, Object> entry : map.entrySet()) {
                    list.add(new SortableObject(entry.getKey(), entry.getValue()));
                }
                Collections.sort(list, new DescendingComparator());
                results = stripVersions(list);
                break;
            }
            default:
                // just do nothing, and return objects as is
                results.addAll(historyTable.row(key).values());
                break;
        }
        return results;
    }

    /**
     * Returns oldest object for specified key, based on version info passed in
     * @param key
     * @return
     */
    public Object getOldest(@NonNull Object key) {
        return get(key, TargetVersion.OLDEST);
    }

    /**
     * Returns latest object for specified key, based on version info passed in
     *
     * @param key
     * @return
     */
    public Object getLatest(@NonNull Object key) {
        return get(key, TargetVersion.LATEST);
    }

    /**
     * This method stores some object along with it's version info
     * @param key
     * @param version
     * @param object
     */
    public void put(@NonNull Object key, Pair<Integer, Integer> version, @NonNull Object object) {
        historyTable.put(key, version, object);
    }

    /**
     * This method removes everything from storage table.
     *
     *
     */
    protected synchronized void wipeStorage() {
        historyTable.clear();
    }

    /**
     * This is a sorting helper method, just strips version info before pushing out stored objects
     * @param objects
     * @return
     */
    protected List<Object> stripVersions(@NonNull List<SortableObject> objects) {
        List<Object> result = new ArrayList<>();
        for (int x = 0; x < objects.size(); x++ ) {
            result.add(objects.get(x).getObject());
        }
        return result;
    }

    /**
     * This method returns number of keys registered within storage.
     *
     * @return
     */
    public int numberOfKeys() {
        return historyTable.rowKeySet().size();
    }

    @Data
    private static class SortableObject {
        @Setter @NonNull private Pair<Integer, Integer> version;
        @Getter @NonNull private Object object;
    }

    private class AscendingComparator implements Comparator<SortableObject> {

        @Override
        public int compare(SortableObject o1, SortableObject o2) {
            // at first we compare major versions. if they match we compare minor versions
            if (o1.getVersion().getFirst().equals(o2.getVersion().getFirst())) {
                // major versions match, we compare minor versions
                return Integer.compare(o1.getVersion().getSecond(), o2.getVersion().getSecond());
            } else {
                // major versions mismatch, so we compare major versions
                return Integer.compare(o1.getVersion().getFirst(), o2.getVersion().getFirst());
            }
        }
    }

    private class DescendingComparator implements Comparator<SortableObject> {

        @Override
        public int compare(SortableObject o1, SortableObject o2) {
            // at first we compare major versions. if they match we compare minor versions
            if (o2.getVersion().getFirst().equals(o1.getVersion().getFirst())) {
                // major versions match, we compare minor versions
                return Integer.compare(o2.getVersion().getSecond(), o1.getVersion().getSecond());
            } else {
                // major versions mismatch, so we compare major versions
                return Integer.compare(o2.getVersion().getFirst(), o1.getVersion().getFirst());
            }
        }
    }
}
