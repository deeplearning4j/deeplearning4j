package org.datavec.dataframe.util;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntCollection;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

import java.util.Set;

/**
 * A map that supports reversible key value pairs of int-String
 */
public class DictionaryMap {

  private final Int2ObjectMap<String> keyToValue = new Int2ObjectOpenHashMap<>();

  private final Object2IntMap<String> valueToKey = new Object2IntOpenHashMap<>();

  public DictionaryMap() {
    super();
    valueToKey.defaultReturnValue(-1);
  }

  /**
   * Returns a new DictionaryMap that is a deep copy of the original
   */
  public DictionaryMap(DictionaryMap original) {
    for (Int2ObjectMap.Entry<String> entry: original.keyToValue.int2ObjectEntrySet()) {
      keyToValue.put(entry.getIntKey(), entry.getValue());
      valueToKey.put(entry.getValue(), entry.getIntKey());
    }
    valueToKey.defaultReturnValue(-1);
  }

  public void put(int key, String value) {
    keyToValue.put(key, value);
    valueToKey.put(value, key);
  }

  public String get(int key) {
    return keyToValue.get(key);
  }

  public int get(String value) {

    return valueToKey.getInt(value);
  }

  public void remove(short key) {
    String value = keyToValue.remove(key);
    valueToKey.remove(value);
  }

  public void remove(String value) {
    int key = valueToKey.remove(value);
    keyToValue.remove(key);
  }

  public void clear() {
    keyToValue.clear();
    valueToKey.clear();
  }

  public boolean contains(String stringValue) {
    return valueToKey.containsKey(stringValue);
  }

  public int size() {
    return categories().size();
  }

  public Set<String> categories() {
    return valueToKey.keySet();
  }

  /**
   * Returns the strings in the dictionary as an array in order of the numeric key
   */
  public String[] categoryArray() {
    return keyToValue.values().toArray(new String[size()]);
  }

  public IntCollection values() {
    return valueToKey.values();
  }

  public Int2ObjectMap<String> keyToValueMap() {
    return keyToValue;
  }

  public Object2IntMap<String> valueToKeyMap() {
    return valueToKey;
  }
}