package org.deeplearning4j.word2vec.iterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.util.Window;
import org.deeplearning4j.word2vec.util.Windows;


public class Word2VecDataSetIteratorImpl implements Word2VecDataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8127389549285313497L;
	private transient Iterator<File> iter;
	private List<String> labels;
	private List<String> cache = new ArrayList<String>();
	private List<Window> windows = new ArrayList<Window>();
	private int batchSize;
	private Word2Vec vec;
	private String path;



	@SuppressWarnings("unchecked")
	public Word2VecDataSetIteratorImpl(String path,List<String> labels,int batchSize,Word2Vec vec) {
		this.iter = FileUtils.iterateFiles(new File(path), null, true);
		this.labels = labels;
		this.path = path;
		this.batchSize = batchSize;
		this.vec = vec;
	}

	@Override
	public boolean hasNext() {
		return iter.hasNext() && cache.isEmpty() && windows.isEmpty();
	}

	@Override
	public List<Window> next() {
		List<Window> nextDataSet = new ArrayList<Window>();
		if(!windows.isEmpty()) {
			if(windows.size() < batchSize) {
				addWhatsLeft(nextDataSet);
				addFromSentenceCache(nextDataSet);

			}
			else {
				addUpToBatch(nextDataSet);
				return toDataSet(nextDataSet);

			}
		}
		
		else  {
			return handleWhileNeeded(nextDataSet);
		}
	


		return toDataSet(nextDataSet);
	}

	private List<Window> handleWhileNeeded(List<Window> nextDataSet) {
		addFromSentenceCache(nextDataSet);

		while(nextDataSet.size() < batchSize) {
			if(!iter.hasNext())
				return toDataSet(nextDataSet);
			if(cache.isEmpty()) {
				try {
					populateSentenceCache();
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
			else 
				addFromSentenceCache(nextDataSet);


		}
		
		return toDataSet(nextDataSet);
	}
	


	private List<Window> toDataSet(List<Window> windows) {
		return new ArrayList<Window>(windows);
	}


	private void populateSentenceCache() throws IOException {
		if(!iter.hasNext()) {
			throw new IllegalStateException("No more files left");
		}
		File next = iter.next();
		LineIterator lines = FileUtils.lineIterator(next);
		while(lines.hasNext()) {
			String line = lines.nextLine();
			if(!line.trim().isEmpty()) {
				cache.add(line);
			}
		}
	}


	private void addFromSentenceCache(List<Window> nextDataSet) {
		while(!cache.isEmpty() && nextDataSet.size() < batchSize) {

			String next = cache.remove(0);
			List<Window> nextWindows = Windows.windows(next);
			while(nextDataSet.size() < batchSize && !nextWindows.isEmpty()) {
				Window nextWindow = nextWindows.remove(0);
				nextDataSet.add(nextWindow);
			}
			//add left overs
			windows.addAll(nextWindows);


		}
	}


	private void addUpToBatch(List<Window> addTo) {
		for(int i = 0; i < batchSize; i++) {
			Window w = windows.remove(0);
			addTo.add(w);

		}
	}

	private void addWhatsLeft(List<Window> addTo) {
		for(int i = 0; i < windows.size(); i++) {
			Window w = windows.remove(0);
			addTo.add(w);
		}
	}


	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int inputColumns() {
		return vec.getLayerSize() * vec.getWindow();
	}

	@Override
	public int totalOutcomes() {
		return labels.size();
	}

	@SuppressWarnings("unchecked")
	@Override
	public void reset() {
		this.iter = FileUtils.iterateFiles(new File(path), null, true);
		this.windows.clear();
		this.cache.clear();

	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int numExamples() {
		// TODO Auto-generated method stub
		return 0;
	}



}
