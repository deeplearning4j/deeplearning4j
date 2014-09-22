package org.deeplearning4j.models.word2vec;

import java.io.InputStream;
import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

public class StreamWork implements Serializable {
	private InputStream is;
	private AtomicInteger count = new AtomicInteger(0);


	public StreamWork(InputStream is, AtomicInteger count) {
		super();
		this.is = is;
		this.count = count;
	}
	public InputStream getIs() {
		return is;
	}
	public void setIs(InputStream is) {
		this.is = is;
	}
	public AtomicInteger getCount() {
		return count;
	}
	public void setCount(AtomicInteger count) {
		this.count = count;
	}
	public void countDown() {
		count.decrementAndGet();

	}




}
