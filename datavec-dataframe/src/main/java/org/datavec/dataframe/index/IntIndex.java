package org.datavec.dataframe.index;

import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.Selection;
import it.unimi.dsi.fastutil.ints.Int2ObjectAVLTreeMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectSortedMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;

import java.util.Comparator;

/**
 * An index for four-byte integer and integer backed columns (date, category, time)
 */
public class IntIndex {

  private final Int2ObjectAVLTreeMap<IntArrayList> index;

  public IntIndex(IntColumn column) {
    int sizeEstimate = Integer.min(1_000_000, column.size() / 100);
    Int2ObjectOpenHashMap<IntArrayList> tempMap = new Int2ObjectOpenHashMap<>(sizeEstimate);
    for (int i = 0; i < column.size(); i++) {
      int value = column.get(i);
      IntArrayList recordIds = tempMap.get(value);
      if (recordIds == null) {
        recordIds = new IntArrayList();
        recordIds.add(i);
        tempMap.trim();
        tempMap.put(value, recordIds);
      } else {
        recordIds.add(i);
      }
    }
    index = new Int2ObjectAVLTreeMap<>(tempMap);
  }

  private final static Comparator<int[]> intArrayComparator = new Comparator<int[]>() {
    public int compare(int[] a, int[] b) {
      return Integer.compare(a[1], b[1]);
    }
  };

  /**
   * Returns a bitmap containing row numbers of all cells matching the given int
   *
   * @param value This is a 'key' from the index perspective, meaning it is a value from the standpoint of the column
   */
  public Selection get(int value) {
    Selection selection = new BitmapBackedSelection();
    IntArrayList list = index.get(value);
    addAllToSelection(list, selection);
    return selection;
  }

  public Selection atLeast(int value) {
    Selection selection = new BitmapBackedSelection();
    Int2ObjectSortedMap<IntArrayList> tail = index.tailMap(value);
    for (IntArrayList keys : tail.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  public Selection greaterThan(int value) {
    Selection selection = new BitmapBackedSelection();
    Int2ObjectSortedMap<IntArrayList> tail = index.tailMap(value + 1);
    for (IntArrayList keys : tail.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  public Selection atMost(int value) {
    Selection selection = new BitmapBackedSelection();
    Int2ObjectSortedMap<IntArrayList> head = index.headMap(value + 1);  // we add 1 to get values equal to the arg
    for (IntArrayList keys : head.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  public Selection lessThan(int value) {
    Selection selection = new BitmapBackedSelection();
    Int2ObjectSortedMap<IntArrayList> head = index.headMap(value);  // we add 1 to get values equal to the arg
    for (IntArrayList keys : head.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  private static void addAllToSelection(IntArrayList tableKeys, Selection selection) {
    for (int i : tableKeys) {
      selection.add(i);
    }
  }
}