package com.ccc.deeplearning.word2vec.viterbi;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import com.ccc.deeplearning.nn.Persistable;
import com.ccc.deeplearning.util.SerializationUtils;
@SuppressWarnings({"rawtypes","unchecked"})
public class Index implements Serializable,Persistable {

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

	@Override
	public void write(OutputStream os) {
		try {
			ObjectOutputStream os2 = new ObjectOutputStream(os);
			os2.writeObject(this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}

	@Override
	public void load(InputStream is) {
		try {
			Index i = (Index) new ObjectInputStream(is).readObject();
			this.indexes = i.indexes;
			this.objects = i.objects;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}