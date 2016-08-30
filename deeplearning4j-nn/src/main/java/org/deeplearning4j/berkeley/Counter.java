/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.berkeley;


import java.io.Serializable;
import java.text.NumberFormat;
import java.util.*;
import java.util.Map.Entry;


/**
 * A map from objects to doubles. Includes convenience methods for getting,
 * setting, and incrementing element counts. Objects not in the counter will
 * return a count of zero. The counter is backed by a HashMap (unless specified
 * otherwise with the MapFactory constructor).
 * 
 * @author Dan Klein
 */
public class Counter<E> implements Serializable {
	private static final long serialVersionUID = 1L;
	Map<E, Double> entries;
	boolean dirty = true;
	double cacheTotal = 0.0;
	MapFactory<E, Double> mf;
	double deflt = 0.0;

	public double getDeflt() {
		return deflt;
	}

	public void setDeflt(double deflt) {
		this.deflt = deflt;
	}

	/**
	 * The elements in the counter.
	 * 
	 * @return applyTransformToDestination of keys
	 */
	public Set<E> keySet() {
		return entries.keySet();
	}

	public Set<Entry<E, Double>> entrySet() {
		return entries.entrySet();
	}

	/**
	 * The number of entries in the counter (not the total count -- use
	 * totalCount() instead).
	 */
	public int size() {
		return entries.size();
	}

	/**
	 * True if there are no entries in the counter (false does not mean
	 * totalCount > 0)
	 */
	public boolean isEmpty() {
		return size() == 0;
	}

	/**
	 * Returns whether the counter contains the given key. Note that this is the
	 * way to distinguish keys which are in the counter with count zero, and
	 * those which are not in the counter (and will therefore return count zero
	 * from getCount().
	 * 
	 * @param key
	 * @return whether the counter contains the key
	 */
	public boolean containsKey(E key) {
		return entries.containsKey(key);
	}

	/**
	 * Get the count of the element, or zero if the element is not in the
	 * counter.
	 * 
	 * @param key
	 * @return
	 */
	public double getCount(E key) {
		Double value = entries.get(key);
		if (value == null) return deflt;
		return value;
	}  

	/**
	 * I know, I know, this should be wrapped in a Distribution class, but it's
	 * such a common use...why not. Returns the MLE prob. Assumes all the counts
	 * are >= 0.0 and totalCount > 0.0. If the latter is false, return 0.0 (i.e.
	 * 0/0 == 0)
	 * 
	 * @author Aria
	 * @param key
	 * @return MLE prob of the key
	 */
	public double getProbability(E key) {
		double count = getCount(key);
		double total = totalCount();
		if (total < 0.0) {
			throw new RuntimeException("Can't call getProbability() with totalCount < 0.0");
		}
		return total > 0.0 ? count / total : 0.0;
	}

	/**
	 * Destructively normalize this Counter in place.
	 */
	public void normalize() {
		double totalCount = totalCount();
		for (E key : keySet()) {
			setCount(key, getCount(key) / totalCount);
		}
		dirty = true;
	}

	/**
	 * Set the count for the given key, clobbering any previous count.
	 * 
	 * @param key
	 * @param count
	 */
	public void setCount(E key, double count) {
		entries.put(key, count);
		dirty = true;
	}

	/**
	 * Set the count for the given key if it is larger than the previous one;
	 * 
	 * @param key
	 * @param count
	 */
	public void put(E key, double count, boolean keepHigher) {
		if (keepHigher && entries.containsKey(key)) {
			double oldCount = entries.get(key);
			if (count > oldCount) {
				entries.put(key, count);
			}
		} else {
			entries.put(key, count);
		}
		dirty = true;
	}

	/**
	 * Will return a sample from the counter, will throw exception if any of the
	 * counts are < 0.0 or if the totalCount() <= 0.0
	 * 
	 * @return
	 * 
	 * @author aria42
	 */
	public E sample(Random rand) {
		double total = totalCount();
		if (total <= 0.0) {
			throw new RuntimeException(String.format(
					"Attempting to sample() with totalCount() %.3f%n", total));
		}
		double sum = 0.0;
		double r = rand.nextDouble();
		for (Entry<E, Double> entry : entries.entrySet()) {
			double count = entry.getValue();
			double frac = count / total;
			sum += frac;
			if (r < sum) {
				return entry.getKey();
			}
		}
		throw new IllegalStateException("Shoudl've have returned a sample by now....");
	}

	/**
	 * Will return a sample from the counter, will throw exception if any of the
	 * counts are < 0.0 or if the totalCount() <= 0.0
	 *
	 * @return
	 *
	 * @author aria42
	 */
	public E sample() {
		return sample(new Random());
	}

	public void removeKey(E key) {
		setCount(key, 0.0);
		dirty = true;
		removeKeyFromEntries(key);
	}

	/**
	 * @param key
	 */
	protected void removeKeyFromEntries(E key) {
		entries.remove(key);
	}

	/**
	 * Set's the key's count to the maximum of the current count and val. Always
	 * sets to val if key is not yet present.
	 *
	 * @param key
	 * @param val
	 */
	public void setMaxCount(E key, double val) {
		Double value = entries.get(key);
		if (value == null || val > value) {
			setCount(key, val);

			dirty = true;
		}
	}

	/**
	 * Set's the key's count to the minimum of the current count and val. Always
	 * sets to val if key is not yet present.
	 *
	 * @param key
	 * @param val
	 */
	public void setMinCount(E key, double val) {
		Double value = entries.get(key);
		if (value == null || val < value) {
			setCount(key, val);

			dirty = true;
		}
	}

	/**
	 * Increment a key's count by the given amount.
	 *
	 * @param key
	 * @param increment
	 */
	public double incrementCount(E key, double increment) {
	  double newVal = getCount(key) + increment;
		setCount(key, newVal);
		dirty = true;
		return newVal;
	}

	/**
	 * Increment each element in a given collection by a given amount.
	 */
	public void incrementAll(Collection<? extends E> collection, double count) {
		for (E key : collection) {
			incrementCount(key, count);
		}
		dirty = true;
	}

	public <T extends E> void incrementAll(Counter<T> counter) {
		for (T key : counter.keySet()) {
			double count = counter.getCount(key);
			incrementCount(key, count);
		}
		dirty = true;
	}

	/**
	 * Finds the total of all counts in the counter. This implementation
	 * iterates through the entire counter every time this method is called.
	 *
	 * @return the counter's total
	 */
	public double totalCount() {
		if (!dirty) {
			return cacheTotal;
		}
		double total = 0.0;
		for (Entry<E, Double> entry : entries.entrySet()) {
			total += entry.getValue();
		}
		cacheTotal = total;
		dirty = false;
		return total;
	}

	public List<E> getSortedKeys() {
		PriorityQueue<E> pq = this.asPriorityQueue();
		List<E> keys = new ArrayList<>();
		while (pq.hasNext()) {
			keys.add(pq.next());
		}
		return keys;
	}

	/**
	 * Finds the key with maximum count. This is a linear operation, and ties
	 * are broken arbitrarily.
	 *
	 * @return a key with minumum count
	 */
	public E argMax() {
		double maxCount = Double.NEGATIVE_INFINITY;
		E maxKey = null;
		for (Entry<E, Double> entry : entries.entrySet()) {
			if (entry.getValue() > maxCount || maxKey == null) {
				maxKey = entry.getKey();
				maxCount = entry.getValue();
			}
		}
		return maxKey;
	}

	public double min() {
		return maxMinHelp(false);
	}

	public double max() {
		return maxMinHelp(true);
	}

	private double maxMinHelp(boolean max) {
		double maxCount = max ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;

		for (Entry<E, Double> entry : entries.entrySet()) {
			if ((max && entry.getValue() > maxCount)
					|| (!max && entry.getValue() < maxCount)) {

				maxCount = entry.getValue();
			}
		}
		return maxCount;
	}

	/**
	 * Returns a string representation with the keys ordered by decreasing
	 * counts.
	 *
	 * @return string representation
	 */
	@Override
	public String toString() {
		return toString(keySet().size());
	}

	public String toStringSortedByKeys() {
		StringBuilder sb = new StringBuilder("[");

		NumberFormat f = NumberFormat.getInstance();
		f.setMaximumFractionDigits(5);
		int numKeysPrinted = 0;
		for (E element : new TreeSet<>(keySet())) {

			sb.append(element.toString());
			sb.append(" : ");
			sb.append(f.format(getCount(element)));
			if (numKeysPrinted < size() - 1) sb.append(", ");
			numKeysPrinted++;
		}
		if (numKeysPrinted < size()) sb.append("...");
		sb.append("]");
		return sb.toString();
	}

	/**
	 * Returns a string representation which includes no more than the
	 * maxKeysToPrint elements with largest counts.
	 *
	 * @param maxKeysToPrint
	 * @return partial string representation
	 */
	public String toString(int maxKeysToPrint) {
		return asPriorityQueue().toString(maxKeysToPrint, false);
	}

	/**
	 * Returns a string representation which includes no more than the
	 * maxKeysToPrint elements with largest counts and optionally prints
	 * one element per line.
	 *
	 * @param maxKeysToPrint
	 * @return partial string representation
	 */
	public String toString(int maxKeysToPrint, boolean multiline) {
		return asPriorityQueue().toString(maxKeysToPrint, multiline);
	}

	/**
	 * Builds a priority queue whose elements are the counter's elements, and
	 * whose priorities are those elements' counts in the counter.
	 */
	public PriorityQueue<E> asPriorityQueue() {
		PriorityQueue<E> pq = new PriorityQueue<>(entries.size());
		for (Entry<E, Double> entry : entries.entrySet()) {
			pq.add(entry.getKey(), entry.getValue());
		}
		return pq;
	}

	/**
	 * Warning: all priorities are the negative of their counts in the counter
	 * here
	 *
	 * @return
	 */
	public PriorityQueue<E> asMinPriorityQueue() {
		PriorityQueue<E> pq = new PriorityQueue<>(entries.size());
		for (Entry<E, Double> entry : entries.entrySet()) {
			pq.add(entry.getKey(), -entry.getValue());
		}
		return pq;
	}

	public Counter() {
		this(false);
	}

	public Counter(boolean identityHashMap) {
		this(identityHashMap ? new MapFactory.IdentityHashMapFactory<E, Double>()
				: new MapFactory.HashMapFactory<E, Double>());
	}

	public Counter(MapFactory<E, Double> mf) {
		this.mf = mf;
		entries = mf.buildMap();
	}

	public Counter(Map<? extends E, Double> mapCounts) {
		this(false);
		this.entries = new HashMap<>();
		for (Entry<? extends E, Double> entry : mapCounts.entrySet()) {
			incrementCount(entry.getKey(), entry.getValue());
		}
	}

	public Counter(Counter<? extends E> counter) {
		this();
		incrementAll(counter);
	}

	public Counter(Collection<? extends E> collection) {
		this();
		incrementAll(collection, 1.0);
	}

	public void pruneKeysBelowThreshold(double cutoff) {
		Iterator<E> it = entries.keySet().iterator();
		while (it.hasNext()) {
			E key = it.next();
			double val = entries.get(key);
			if (val < cutoff) {
				it.remove();
			}
		}
		dirty = true;
	}

	public Set<Entry<E, Double>> getEntrySet() {
		return entries.entrySet();
	}

	public boolean isEqualTo(Counter<E> counter) {
		boolean tmp = true;
		Counter<E> bigger = counter.size() > size() ? counter : this;
		for (E e : bigger.keySet()) {
			tmp &= counter.getCount(e) == getCount(e);
		}
		return tmp;
	}

	public static void main(String[] args) {
		Counter<String> counter = new Counter<>();
		System.out.println(counter);
		counter.incrementCount("planets", 7);
		System.out.println(counter);
		counter.incrementCount("planets", 1);
		System.out.println(counter);
		counter.setCount("suns", 1);
		System.out.println(counter);
		counter.setCount("aliens", 0);
		System.out.println(counter);
		System.out.println(counter.toString(2));
		System.out.println("Total: " + counter.totalCount());
	}

	public void clear() {
		entries = mf.buildMap();
		dirty = true;
	}

	public void keepTopNKeys(int keepN) {
		keepKeysHelper(keepN, true);
	}

	public void keepBottomNKeys(int keepN) {
		keepKeysHelper(keepN, false);
	}

	private void keepKeysHelper(int keepN, boolean top) {
		Counter<E> tmp = new Counter<>();

		int n = 0;
		for (E e : Iterators.able(top ? asPriorityQueue() : asMinPriorityQueue())) {

			if (n <= keepN) tmp.setCount(e, getCount(e));
			n++;

		}
		clear();
		incrementAll(tmp);
		dirty = true;

	}

	/**
	 * Sets all counts to the given value, but does not remove any keys
	 */
	public void setAllCounts(double val) {
		for (E e : keySet()) {
			setCount(e, val);
		}

	}

	public double dotProduct(Counter<E> other) {
		double sum = 0.0;
		for (Entry<E, Double> entry : getEntrySet()) {
			final double otherCount = other.getCount(entry.getKey());
			if (otherCount == 0.0) continue;
			final double value = entry.getValue();
			if (value == 0.0) continue;
			sum += value * otherCount;

		}
		return sum;
	}

	public void scale(double c) {

		for (Entry<E, Double> entry : getEntrySet()) {
			entry.setValue(entry.getValue() * c);
		}

	}

	public Counter<E> scaledClone(double c) {
		Counter<E> newCounter = new Counter<>();

		for (Entry<E, Double> entry : getEntrySet()) {
			newCounter.setCount(entry.getKey(), entry.getValue() * c);
		}

		return newCounter;
	}

	public Counter<E> difference(Counter<E> counter) {
		Counter<E> clone = new Counter<>(this);
		for (E key : counter.keySet()) {
			double count = counter.getCount(key);
			clone.incrementCount(key, -1 * count);
		}
		return clone;
	}

	public Counter<E> toLogSpace() {
		Counter<E> newCounter = new Counter<>(this);
		for (E key : newCounter.keySet()) {
			newCounter.setCount(key, Math.log(getCount(key)));
		}
		return newCounter;
	}

	public boolean approxEquals(Counter<E> other, double tol) {
		for (E key : keySet()) {
			if (Math.abs(getCount(key) - other.getCount(key)) > tol) return false;
		}
		for (E key : other.keySet()) {
			if (Math.abs(getCount(key) - other.getCount(key)) > tol) return false;
		}
		return true;
	}

  public void setDirty(boolean dirty) {
    this.dirty = dirty;
  }

	public String toStringTabSeparated() {
		StringBuilder sb = new StringBuilder();
		for (E key : getSortedKeys()) {
			sb.append(key.toString() + "\t" + getCount(key) + "\n");
		}
		return sb.toString();
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		Counter<?> counter = (Counter<?>) o;

		if (dirty != counter.dirty) return false;
		if (Double.compare(counter.cacheTotal, cacheTotal) != 0) return false;
		if (Double.compare(counter.deflt, deflt) != 0) return false;
		return !(entries != null ? !entries.equals(counter.entries) : counter.entries != null);

	}

	@Override
	public int hashCode() {
		int result;
		long temp;
		result = entries != null ? entries.hashCode() : 0;
		result = 31 * result + (dirty ? 1 : 0);
		temp = Double.doubleToLongBits(cacheTotal);
		result = 31 * result + (int) (temp ^ (temp >>> 32));
		result = 31 * result + (mf != null ? mf.hashCode() : 0);
		temp = Double.doubleToLongBits(deflt);
		result = 31 * result + (int) (temp ^ (temp >>> 32));
		return result;
	}
}
