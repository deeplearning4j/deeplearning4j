package org.deeplearning4j.matrix.jblas.iterativereduce.actor.core;

import java.io.File;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class SerializableFileIter implements Iterator<File>,Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6799392471121716534L;
	private List<File> files;
	private int curr = 0;



	public SerializableFileIter(List<File> files) {
		super();
		this.files = files;
	}

	public synchronized int getCurr() {
		return curr;
	}

	public synchronized void setCurr(int curr) {
		this.curr = curr;
	}

	@Override
	public boolean hasNext() {
		return curr < files.size();
	}

	@Override
	public File next() {
		File ret = files.get(curr);
		curr++;
		return ret;
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

}
