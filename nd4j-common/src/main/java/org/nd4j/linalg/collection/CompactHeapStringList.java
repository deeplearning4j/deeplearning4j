package org.nd4j.linalg.collection;

import java.util.*;

/**
 * A {@code List<String>} that stores all contents in a single char[], to avoid the GC load for a large number of String
 * objects.<br>
 * <p>
 * Some restrictions to be aware of with the current implementation:<br>
 * - The list is intended to be write-once (append only), except for clear() operations. That is: new Strings can be added
 *   at the end, but they cannot be replaced or removed.<br>
 * - There is a limit of a maximum of {@link Integer#MAX_VALUE}/2 = 1073741823 Strings<br>
 * - There is a limit of the maximum total characters of {@link Integer#MAX_VALUE} (i.e., 2147483647 chars). This corresponds
 *   to a maximum of approximately 4GB of Strings.<br>
 *
 * @author Alex Black
 */
public class CompactHeapStringList implements List<String> {
    public static final int DEFAULT_REALLOCATION_BLOCK_SIZE_BYTES = 8 * 1024 * 1024; //8MB
    public static final int DEFAULT_INTEGER_REALLOCATION_BLOCK_SIZE_BYTES = 1024 * 1024; //1MB - 262144 ints, 131k entries

    private final int reallocationBlockSizeBytes;
    private final int reallocationIntegerBlockSizeBytes;
    private int usedCount = 0;
    private int nextDataOffset = 0;
    private char[] data;
    private int[] offsetAndLength;

    public CompactHeapStringList() {
        this(DEFAULT_REALLOCATION_BLOCK_SIZE_BYTES, DEFAULT_INTEGER_REALLOCATION_BLOCK_SIZE_BYTES);
    }

    /**
     *
     * @param reallocationBlockSizeBytes    Number of bytes by which to increase the char[], when allocating a new storage array
     * @param intReallocationBlockSizeBytes Number of bytes by which to increase the int[], when allocating a new storage array
     */
    public CompactHeapStringList(int reallocationBlockSizeBytes, int intReallocationBlockSizeBytes) {
        this.reallocationBlockSizeBytes = reallocationBlockSizeBytes;
        this.reallocationIntegerBlockSizeBytes = intReallocationBlockSizeBytes;

        this.data = new char[this.reallocationBlockSizeBytes / 2];
        this.offsetAndLength = new int[this.reallocationIntegerBlockSizeBytes / 4];
    }

    @Override
    public int size() {
        return usedCount;
    }

    @Override
    public boolean isEmpty() {
        return usedCount == 0;
    }

    @Override
    public boolean contains(Object o) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public Iterator<String> iterator() {
        return new CompactHeapStringListIterator();
    }

    @Override
    public String[] toArray() {
        String[] str = new String[usedCount];
        for (int i = 0; i < usedCount; i++) {
            str[i] = get(i);
        }
        return str;
    }

    @Override
    public <T> T[] toArray(T[] a) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public boolean add(String s) {
        int length = s.length();
        //3 possibilities:
        //(a) doesn't fit in char[]
        //(b) doesn't fit in int[]
        //(c) fits OK in both

        if (nextDataOffset + length > data.length) {
            //Allocate new data array, if possible
            if (nextDataOffset > Integer.MAX_VALUE - length) {
                throw new UnsupportedOperationException(
                                "Cannot allocate new data char[]: required array size exceeds Integer.MAX_VALUE");
            }
            int toAdd = Math.max(reallocationBlockSizeBytes / 2, length);
            int newLength = data.length + Math.min(toAdd, Integer.MAX_VALUE - data.length);
            data = Arrays.copyOf(data, newLength);
        }
        if (2 * (usedCount + 1) >= offsetAndLength.length) {
            if (offsetAndLength.length >= Integer.MAX_VALUE - 2) {
                //Should normally never happen
                throw new UnsupportedOperationException(
                                "Cannot allocate new offset int[]: required array size exceeds Integer.MAX_VALUE");
            }
            int newLength = offsetAndLength.length + Math.min(reallocationIntegerBlockSizeBytes / 4,
                            Integer.MAX_VALUE - offsetAndLength.length);
            offsetAndLength = Arrays.copyOf(offsetAndLength, newLength);
        }


        s.getChars(0, length, data, nextDataOffset);
        offsetAndLength[2 * usedCount] = nextDataOffset;
        offsetAndLength[2 * usedCount + 1] = length;
        nextDataOffset += length;
        usedCount++;

        return true;
    }

    @Override
    public boolean remove(Object o) {
        //In principle we *could* do this with array copies
        throw new UnsupportedOperationException("Remove not supported");
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public boolean addAll(Collection<? extends String> c) {
        for (String s : c) {
            add(s);
        }
        return c.size() > 0;
    }

    @Override
    public boolean addAll(int index, Collection<? extends String> c) {
        //This is conceivably possible with array copies and adjusting the indices
        throw new UnsupportedOperationException("Add all at specified index: Not supported");
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException("Remove all: Not supported");
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException("Retain all: Not supported");
    }

    @Override
    public void clear() {
        usedCount = 0;
        nextDataOffset = 0;
        data = new char[reallocationBlockSizeBytes / 2];
        offsetAndLength = new int[reallocationIntegerBlockSizeBytes / 4];
    }

    @Override
    public String get(int index) {
        if (index >= usedCount) {
            throw new IllegalArgumentException("Invalid index: " + index + " >= size(). Size = " + usedCount);
        }
        int offset = offsetAndLength[2 * index];
        int length = offsetAndLength[2 * index + 1];
        return new String(data, offset, length);
    }

    @Override
    public String set(int index, String element) {
        //This *could* be done with array copy ops...
        throw new UnsupportedOperationException(
                        "Set specified index: not supported due to serialized storage structure");
    }

    @Override
    public void add(int index, String element) {
        //This *could* be done with array copy ops...
        throw new UnsupportedOperationException(
                        "Set specified index: not supported due to serialized storage structure");
    }

    @Override
    public String remove(int index) {
        throw new UnsupportedOperationException("Remove: not supported");
    }

    @Override
    public int indexOf(Object o) {
        if (!(o instanceof String)) {
            return -1;
        }

        String str = (String) o;
        char[] ch = str.toCharArray();


        for (int i = 0; i < usedCount; i++) {
            if (offsetAndLength[2 * i + 1] != ch.length) {
                //Can't be this one: lengths differ
                continue;
            }
            int offset = offsetAndLength[2 * i];

            boolean matches = true;
            for (int j = 0; j < ch.length; j++) {
                if (data[offset + j] != ch[j]) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return i;
            }
        }

        return -1;
    }

    @Override
    public int lastIndexOf(Object o) {
        if (!(o instanceof String)) {
            return -1;
        }

        String str = (String) o;
        char[] ch = str.toCharArray();


        for (int i = usedCount - 1; i >= 0; i--) {
            if (offsetAndLength[2 * i + 1] != ch.length) {
                //Can't be this one: lengths differ
                continue;
            }
            int offset = offsetAndLength[2 * i];

            boolean matches = true;
            for (int j = 0; j < ch.length; j++) {
                if (data[offset + j] != ch[j]) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return i;
            }
        }

        return -1;
    }

    @Override
    public ListIterator<String> listIterator() {
        return new CompactHeapStringListIterator();
    }

    @Override
    public ListIterator<String> listIterator(int index) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<String> subList(int fromIndex, int toIndex) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (!(o instanceof List))
            return false;

        ListIterator<String> e1 = listIterator();
        ListIterator<?> e2 = ((List<?>) o).listIterator();
        while (e1.hasNext() && e2.hasNext()) {
            String o1 = e1.next();
            Object o2 = e2.next();
            if (!(o1 == null ? o2 == null : o1.equals(o2)))
                return false;
        }
        return !(e1.hasNext() || e2.hasNext());
    }

    private class CompactHeapStringListIterator implements Iterator<String>, ListIterator<String> {
        private int currIdx = 0;

        @Override
        public boolean hasNext() {
            return currIdx < usedCount;
        }

        @Override
        public String next() {
            if (!hasNext()) {
                throw new NoSuchElementException("No next element");
            }
            return get(currIdx++);
        }

        @Override
        public boolean hasPrevious() {
            return currIdx > 0;
        }

        @Override
        public String previous() {
            if (!hasPrevious()) {
                throw new NoSuchElementException();
            }
            return get(currIdx--);
        }

        @Override
        public int nextIndex() {
            return currIdx;
        }

        @Override
        public int previousIndex() {
            return currIdx;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void set(String s) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void add(String s) {
            throw new UnsupportedOperationException();
        }
    }
}
