package org.datavec.dataframe.index;

import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.util.BitmapBackedSelection;
import org.datavec.dataframe.util.Selection;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.longs.Long2ObjectAVLTreeMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectSortedMap;

/**
 * An index for eight-byte long and long backed columns (datetime)
 */
public class LongIndex {

  private final Long2ObjectAVLTreeMap<IntArrayList> index;

  public LongIndex(LongColumn column) {
    int sizeEstimate = Integer.min(1_000_000, column.size() / 100);
    Long2ObjectOpenHashMap<IntArrayList> tempMap = new Long2ObjectOpenHashMap<>(sizeEstimate);
    for (int i = 0; i < column.size(); i++) {
      long value = column.get(i);
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
    index = new Long2ObjectAVLTreeMap<>(tempMap);
  }

  /**
   * Returns a bitmap containing row numbers of all cells matching the given long
   *
   * @param value This is a 'key' from the index perspective, meaning it is a value from the standpoint of the column
   */
  public Selection get(long value) {
    Selection selection = new BitmapBackedSelection();
    IntArrayList list = index.get(value);
    addAllToSelection(list, selection);
    return selection;
  }

  public Selection atLeast(int value) {
    Selection selection = new BitmapBackedSelection();
    Long2ObjectSortedMap<IntArrayList> tail = index.tailMap(value);
    for (IntArrayList keys : tail.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  public Selection greaterThan(int value) {
    Selection selection = new BitmapBackedSelection();
    Long2ObjectSortedMap<IntArrayList> tail = index.tailMap(value + 1);
    for (IntArrayList keys : tail.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  public Selection atMost(int value) {
    Selection selection = new BitmapBackedSelection();
    Long2ObjectSortedMap<IntArrayList> head = index.headMap(value + 1);  // we add 1 to get values equal to the arg
    for (IntArrayList keys : head.values()) {
      addAllToSelection(keys, selection);
    }
    return selection;
  }

  public Selection lessThan(int value) {
    Selection selection = new BitmapBackedSelection();
    Long2ObjectSortedMap<IntArrayList> head = index.headMap(value);  // we add 1 to get values equal to the arg
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