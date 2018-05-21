package org.nd4j.linalg.io;

import org.nd4j.linalg.util.MultiValueMap;

import java.io.Serializable;
import java.util.*;
import java.util.Map.Entry;


public abstract class CollectionUtils {
    public CollectionUtils() {}

    public static boolean isEmpty(Collection collection) {
        return collection == null || collection.isEmpty();
    }

    public static boolean isEmpty(Map map) {
        return map == null || map.isEmpty();
    }

    public static List arrayToList(Object source) {
        return Arrays.asList(ObjectUtils.toObjectArray(source));
    }

    public static void mergeArrayIntoCollection(Object array, Collection collection) {
        if (collection == null) {
            throw new IllegalArgumentException("Collection must not be null");
        } else {
            Object[] arr = ObjectUtils.toObjectArray(array);
            Object[] arr$ = arr;
            int len$ = arr.length;

            for (int i$ = 0; i$ < len$; ++i$) {
                Object elem = arr$[i$];
                collection.add(elem);
            }

        }
    }

    public static void mergePropertiesIntoMap(Properties props, Map map) {
        if (map == null) {
            throw new IllegalArgumentException("Map must not be null");
        } else {
            String key;
            Object value;
            if (props != null) {
                for (Enumeration en = props.propertyNames(); en.hasMoreElements(); map.put(key, value)) {
                    key = (String) en.nextElement();
                    value = props.getProperty(key);
                    if (value == null) {
                        value = props.get(key);
                    }
                }
            }

        }
    }

    public static boolean contains(Iterator iterator, Object element) {
        if (iterator != null) {
            while (iterator.hasNext()) {
                Object candidate = iterator.next();
                if (ObjectUtils.nullSafeEquals(candidate, element)) {
                    return true;
                }
            }
        }

        return false;
    }

    public static boolean contains(Enumeration enumeration, Object element) {
        if (enumeration != null) {
            while (enumeration.hasMoreElements()) {
                Object candidate = enumeration.nextElement();
                if (ObjectUtils.nullSafeEquals(candidate, element)) {
                    return true;
                }
            }
        }

        return false;
    }

    public static boolean containsInstance(Collection collection, Object element) {
        if (collection != null) {
            Iterator i$ = collection.iterator();

            while (i$.hasNext()) {
                Object candidate = i$.next();
                if (candidate == element) {
                    return true;
                }
            }
        }

        return false;
    }

    public static boolean containsAny(Collection source, Collection candidates) {
        if (!isEmpty(source) && !isEmpty(candidates)) {
            Iterator i$ = candidates.iterator();

            Object candidate;
            do {
                if (!i$.hasNext()) {
                    return false;
                }

                candidate = i$.next();
            } while (!source.contains(candidate));

            return true;
        } else {
            return false;
        }
    }

    public static Object findFirstMatch(Collection source, Collection candidates) {
        if (!isEmpty(source) && !isEmpty(candidates)) {
            Iterator i$ = candidates.iterator();

            Object candidate;
            do {
                if (!i$.hasNext()) {
                    return null;
                }

                candidate = i$.next();
            } while (!source.contains(candidate));

            return candidate;
        } else {
            return null;
        }
    }

    public static <T> T findValueOfType(Collection<?> collection, Class<T> type) {
        if (isEmpty((Collection) collection)) {
            return null;
        } else {
            Object value = null;
            Iterator i$ = collection.iterator();

            while (i$.hasNext()) {
                Object element = i$.next();
                if (type == null || type.isInstance(element)) {
                    if (value != null) {
                        return null;
                    }

                    value = element;
                }
            }

            return (T) value;
        }
    }

    public static Object findValueOfType(Collection<?> collection, Class<?>[] types) {
        if (!isEmpty((Collection) collection) && !ObjectUtils.isEmpty(types)) {
            Class[] arr$ = types;
            int len$ = types.length;

            for (int i$ = 0; i$ < len$; ++i$) {
                Class type = arr$[i$];
                Object value = findValueOfType(collection, type);
                if (value != null) {
                    return value;
                }
            }

            return null;
        } else {
            return null;
        }
    }

    public static boolean hasUniqueObject(Collection collection) {
        if (isEmpty(collection)) {
            return false;
        } else {
            boolean hasCandidate = false;
            Object candidate = null;
            Iterator i$ = collection.iterator();

            while (i$.hasNext()) {
                Object elem = i$.next();
                if (!hasCandidate) {
                    hasCandidate = true;
                    candidate = elem;
                } else if (candidate != elem) {
                    return false;
                }
            }

            return true;
        }
    }

    public static Class<?> findCommonElementType(Collection collection) {
        if (isEmpty(collection)) {
            return null;
        } else {
            Class candidate = null;
            Iterator i$ = collection.iterator();

            while (i$.hasNext()) {
                Object val = i$.next();
                if (val != null) {
                    if (candidate == null) {
                        candidate = val.getClass();
                    } else if (candidate != val.getClass()) {
                        return null;
                    }
                }
            }

            return candidate;
        }
    }

    public static <A, E extends A> A[] toArray(Enumeration<E> enumeration, A[] array) {
        ArrayList elements = new ArrayList();

        while (enumeration.hasMoreElements()) {
            elements.add(enumeration.nextElement());
        }

        return (A[]) elements.toArray(array);
    }

    public static <E> Iterator<E> toIterator(Enumeration<E> enumeration) {
        return new CollectionUtils.EnumerationIterator(enumeration);
    }

    public static <K, V> MultiValueMap<K, V> toMultiValueMap(Map<K, List<V>> map) {
        return new CollectionUtils.MultiValueMapAdapter(map);
    }

    public static <K, V> MultiValueMap<K, V> unmodifiableMultiValueMap(MultiValueMap<? extends K, ? extends V> map) {
        Assert.notNull(map, "\'map\' must not be null");
        LinkedHashMap result = new LinkedHashMap(map.size());
        Iterator unmodifiableMap = map.entrySet().iterator();

        while (unmodifiableMap.hasNext()) {
            Entry entry = (Entry) unmodifiableMap.next();
            List values = Collections.unmodifiableList((List) entry.getValue());
            result.put(entry.getKey(), values);
        }

        Map unmodifiableMap1 = Collections.unmodifiableMap(result);
        return toMultiValueMap(unmodifiableMap1);
    }

    private static class MultiValueMapAdapter<K, V> implements MultiValueMap<K, V>, Serializable {
        private final Map<K, List<V>> map;

        public MultiValueMapAdapter(Map<K, List<V>> map) {
            Assert.notNull(map, "\'map\' must not be null");
            this.map = map;
        }

        public void add(K key, V value) {
            List<V> values = this.map.get(key);
            if (values == null) {
                values = new LinkedList<>();
                this.map.put(key, values);
            }

            values.add(value);
        }

        public V getFirst(K key) {
            List values = this.map.get(key);
            return values != null ? (V) values.get(0) : null;
        }

        public void set(K key, V value) {
            LinkedList values = new LinkedList();
            values.add(value);
            this.map.put(key, values);
        }

        public void setAll(Map<K, V> values) {
            Iterator i$ = values.entrySet().iterator();

            while (i$.hasNext()) {
                Entry entry = (Entry) i$.next();
                this.set((K) entry.getKey(), (V) entry.getValue());
            }

        }

        public Map<K, V> toSingleValueMap() {
            LinkedHashMap singleValueMap = new LinkedHashMap(this.map.size());
            Iterator i$ = this.map.entrySet().iterator();

            while (i$.hasNext()) {
                Entry entry = (Entry) i$.next();
                singleValueMap.put(entry.getKey(), ((List) entry.getValue()).get(0));
            }

            return singleValueMap;
        }

        public int size() {
            return this.map.size();
        }

        public boolean isEmpty() {
            return this.map.isEmpty();
        }

        public boolean containsKey(Object key) {
            return this.map.containsKey(key);
        }

        public boolean containsValue(Object value) {
            return this.map.containsValue(value);
        }

        public List<V> get(Object key) {
            return this.map.get(key);
        }

        public List<V> put(K key, List<V> value) {
            return this.map.put(key, value);
        }

        public List<V> remove(Object key) {
            return this.map.remove(key);
        }

        public void putAll(Map<? extends K, ? extends List<V>> m) {
            this.map.putAll(m);
        }

        public void clear() {
            this.map.clear();
        }

        public Set<K> keySet() {
            return this.map.keySet();
        }

        public Collection<List<V>> values() {
            return this.map.values();
        }

        public Set<Entry<K, List<V>>> entrySet() {
            return this.map.entrySet();
        }

        public boolean equals(Object other) {
            return this == other ? true : this.map.equals(other);
        }

        public int hashCode() {
            return this.map.hashCode();
        }

        public String toString() {
            return this.map.toString();
        }
    }

    private static class EnumerationIterator<E> implements Iterator<E> {
        private Enumeration<E> enumeration;

        public EnumerationIterator(Enumeration<E> enumeration) {
            this.enumeration = enumeration;
        }

        public boolean hasNext() {
            return this.enumeration.hasMoreElements();
        }

        public E next() {
            return this.enumeration.nextElement();
        }

        public void remove() throws UnsupportedOperationException {
            throw new UnsupportedOperationException("Not supported");
        }
    }
}
