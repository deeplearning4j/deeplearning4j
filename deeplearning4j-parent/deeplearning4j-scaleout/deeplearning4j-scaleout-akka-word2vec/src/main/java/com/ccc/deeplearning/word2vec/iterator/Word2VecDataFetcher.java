package com.ccc.deeplearning.word2vec.iterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.DataSetFetcher;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.WindowConverter;
import com.ccc.deeplearning.word2vec.util.Windows;

@SuppressWarnings("unchecked")
public class Word2VecDataFetcher implements DataSetFetcher {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3245955804749769475L;
	private Iterator<File> files;
	private Word2Vec vec;
	private static Pattern begin = Pattern.compile("<[A-Z]+>");
	private static Pattern end = Pattern.compile("</[A-Z]+>");
	private List<String> labels = new ArrayList<String>();
	private int batch;
	private List<Window> cache = new ArrayList<Window>();
	private static Logger log = LoggerFactory.getLogger(Word2VecDataFetcher.class);
	private int totalExamples;
	private int minWordFrequency;
	
	public Word2VecDataFetcher(String path,Word2Vec vec,List<String> labels) {
		if(vec == null || labels == null || labels.isEmpty())
			throw new IllegalArgumentException("Unable to initialize due to missing argument or empty label set");
		this.vec = vec;
		this.labels = labels;
	}
	
	public Word2VecDataFetcher(String path,int minWordFrequency) {
		this.minWordFrequency = minWordFrequency;
		collectLabelsAndInitWord2Vec(path);
	}

	private void collectLabelsAndInitWord2Vec(String path) {
		files = FileUtils.iterateFiles(new File(path), null, true);
		vec = new Word2Vec(FileUtils.iterateFiles(new File(path), null, true));
		
		vec.setMinWordFrequency(minWordFrequency);
		Set<String> labels = new HashSet<String>();
		while(files.hasNext()) {
			File next = files.next();
			try {
				LineIterator lines = FileUtils.lineIterator(next);
				if(!lines.hasNext())
					continue;
				for(String line = lines.nextLine(); lines.hasNext();line = lines.nextLine()) {
					//each window is an example
					List<Window> windows = Windows.windows(line);
					totalExamples += windows.size();
					
					vec.addToVocab(line);
					Matcher beginMatcher = begin.matcher(line);
					Matcher endMatcher = end.matcher(line);
					//found pair
					while(beginMatcher.find() && endMatcher.find()) {
						//validate equal: add this as a label if it doesn't exist
						String beginGroup = beginMatcher.group();
						String endGroup = endMatcher.group();
						beginGroup = beginGroup.replace("<","").replace(">","");
						endGroup = endGroup.replace("<","").replace(">","");
						if(beginGroup.equals(endGroup)) {
							labels.add(beginGroup);
						}

					}
				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		
		vec.setup();
		files = FileUtils.iterateFiles(new File(path), null, true);

		//builds vocab and other associated tools
		vec.train();
		//reset file iterator for training on files
		files = FileUtils.iterateFiles(new File(path), null, true);
		//unique labels only: capture everything in a list for index of operations
		this.labels.addAll(labels);
	}


	

	private DataSet fromCache() {
		DoubleMatrix outcomes = null;
		DoubleMatrix input = null;
		input = new DoubleMatrix(batch,vec.getSyn1().columns * vec.getWindow());
		outcomes = new DoubleMatrix(batch,labels.size());
		for(int i = 0; i < batch; i++) {
			input.putRow(i,new DoubleMatrix(WindowConverter.asExample(cache.get(i), vec)));
			outcomes.put(i,labels.indexOf(cache.get(i).getLabel()),1.0);
		}
		return new DataSet(input,outcomes);
		
	}
	
	@Override
	public DataSet next() {
		//pop from cache when possible, or when there's nothing left
		if(cache.size() >= batch || !files.hasNext()) 
			return fromCache();

		

		File f = files.next();
		try {
			LineIterator lines = FileUtils.lineIterator(f);
			DoubleMatrix outcomes = null;
			DoubleMatrix input = null;
			
			while(lines.hasNext()) {
				List<Window> windows = Windows.windows(lines.nextLine());
				
				if(windows.size() < batch) {
					input = new DoubleMatrix(windows.size(),vec.getSyn1().columns * vec.getWindow());
					outcomes = new DoubleMatrix(batch,labels.size());
					for(int i = 0; i < windows.size(); i++) {
						input.putRow(i,new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)));
						outcomes.put(i,labels.indexOf(windows.get(i).getLabel()),1.0);
					}
					return new DataSet(input,outcomes);


				}
				else {
					input = new DoubleMatrix(batch,vec.getSyn1().columns * vec.getWindow());
					outcomes = new DoubleMatrix(batch,labels.size());
					for(int i = 0; i < batch; i++) {
						input.putRow(i,new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)));
						outcomes.put(i,labels.indexOf(windows.get(i).getLabel()),1.0);
					}
					//add left over to cache; need to ensure that only batch rows are returned
					/*
					 * Note that I'm aware of possible concerns for sentence sequencing.
					 * This is a hack right now in place of something
					 * that will be way more elegant in the future.
					 */
					if(windows.size() > batch) {
						List<Window> leftOvers = windows.subList(batch,windows.size());
						cache.addAll(leftOvers);
					}
					return new DataSet(input,outcomes);
				}

			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		return null;
	}

	

	@Override
	public int totalExamples() {
		return totalExamples;
	}

	@Override
	public int inputColumns() {
		return labels.size();
	}

	@Override
	public int totalOutcomes() {
		return labels.size();

	}

	@Override
	public void reset() {
		throw new UnsupportedOperationException();

	}



	@Override
	public int cursor() {
		throw new UnsupportedOperationException();

	}



	@Override
	public boolean hasMore() {
		return files.hasNext() && cache.isEmpty();
	}

	@Override
	public void fetch(int numExamples) {
		this.batch = numExamples;
	}

	public Iterator<File> getFiles() {
		return files;
	}

	public Word2Vec getVec() {
		return vec;
	}

	public static Pattern getBegin() {
		return begin;
	}

	public static Pattern getEnd() {
		return end;
	}

	public List<String> getLabels() {
		return labels;
	}

	public int getBatch() {
		return batch;
	}

	public List<Window> getCache() {
		return cache;
	}



	

}
