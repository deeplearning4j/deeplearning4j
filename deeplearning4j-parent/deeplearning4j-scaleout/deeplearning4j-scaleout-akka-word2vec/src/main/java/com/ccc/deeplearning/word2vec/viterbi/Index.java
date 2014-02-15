package com.ccc.deeplearning.word2vec.viterbi;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
@SuppressWarnings({"rawtypes","unchecked"})
public class Index implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1160629777026141078L;
	List objects = new ArrayList();
	Map<Object,Integer> indexes = new HashMap<Object,Integer>();
	
	
	
	public synchronized boolean add(Object o) {
		Integer index = indexes.get(o);
		if (index == null) {
			index = objects.size();
			objects.add(o);
			indexes.put(o, index);
			return true;
		}
		return false;
	}

	public synchronized int indexOf(Object o) {
		Integer index = indexes.get(o);
		if (index == null) { return -1; }
		else { return index; }
	}

	public synchronized Object get(int i) {
		return objects.get(i);
	}

	public int size() {
		return objects.size();
	}

	public String toString() {
		StringBuilder buff = new StringBuilder("[");
		int sz = objects.size();
		int i;
		for (i = 0; i < sz; i++) {
			Object e = objects.get(i);
			buff.append(i).append("=").append(e);
			if (i < (sz-1)) buff.append(",");
		}
		buff.append("]");
		return buff.toString();

	}
}