package com.ccc.deeplearning.word2vec.iterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.jblas.DoubleMatrix;


import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.DataSetIterator;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.WindowConverter;
import com.ccc.deeplearning.word2vec.util.Windows;

/**
 * Trains a word2vec model and iterates over a file dataset
 * containing text files with sentences such as:
 * W1 W2 <LABEL> SOME POSItIVE EXAMPLE </LABEL> W3 W4 W5
 * @author Adam Gibson
 *
 */
@SuppressWarnings("unchecked")
public class Word2VecDataSetIterator implements DataSetIterator {

	private static final long serialVersionUID = 2397051312760991798L;
	private Iterator<File> files;
	private Word2Vec vec;
	private static Pattern begin = Pattern.compile("<[A-Z]+>");
	private static Pattern end = Pattern.compile("</[A-Z]+>");
	private List<String> labels = new ArrayList<String>();
	private int batch;
	private int numExamples;
	private List<Window> cache = new ArrayList<Window>();
	
	
	public Word2VecDataSetIterator(String path,int batch,int numExamples,Word2Vec vec,List<String> labels) {
		if(vec == null || labels == null || labels.isEmpty())
			throw new IllegalArgumentException("Unable to initialize due to missing argument or empty label set");
		this.batch = batch;
		this.numExamples = numExamples;
		this.vec = vec;
		this.labels = labels;
	}
	
	public Word2VecDataSetIterator(String path,int batch,int numExamples) {
		collectLabelsAndInitWord2Vec(path);
		this.batch = batch;
		this.numExamples = numExamples;
	}

	private void collectLabelsAndInitWord2Vec(String path) {
		files = FileUtils.iterateFiles(new File(path), null, true);
		vec = new Word2Vec();
		Set<String> labels = new HashSet<String>();
		while(files.hasNext()) {
			File next = files.next();
			try {
				LineIterator lines = FileUtils.lineIterator(next);
				for(String line = lines.nextLine(); lines.hasNext();) {
					vec.addSentence(line);
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

		//builds vocab and other associated tools
		vec.train();
		//reset file iterator for training on files
		files = FileUtils.iterateFiles(new File(path), null, true);
		//unique labels only: capture everything in a list for index of operations
		this.labels.addAll(labels);
	}


	@Override
	public boolean hasNext() {
		return files.hasNext();
	}

	private DataSet fromCache() {
		DoubleMatrix outcomes = null;
		DoubleMatrix input = null;
		input = new DoubleMatrix(batch(),vec.getSyn1().columns * vec.getWindow());
		outcomes = new DoubleMatrix(batch(),labels.size());
		for(int i = 0; i < batch(); i++) {
			input.putRow(i,new DoubleMatrix(WindowConverter.asExample(cache.get(i), vec)));
			outcomes.put(i,labels.indexOf(cache.get(i).getLabel()),1.0);
		}
		return new DataSet(input,outcomes);
		
	}
	
	@Override
	public DataSet next() {
		if(cache.size() >= batch()) 
			return fromCache();

		

		File f = files.next();
		try {
			LineIterator lines = FileUtils.lineIterator(f);
			while(lines.hasNext()) {
				List<Window> windows = Windows.windows(lines.nextLine());
				DoubleMatrix outcomes = null;
				DoubleMatrix input = null;
				if(windows.size() < batch()) {
					input = new DoubleMatrix(windows.size(),vec.getSyn1().columns * vec.getWindow());
					outcomes = new DoubleMatrix(batch(),labels.size());
					for(int i = 0; i < windows.size(); i++) {
						input.putRow(i,new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)));
						outcomes.put(i,labels.indexOf(windows.get(i).getLabel()),1.0);
					}
					return new DataSet(input,outcomes);


				}
				else {
					input = new DoubleMatrix(batch(),vec.getSyn1().columns * vec.getWindow());
					outcomes = new DoubleMatrix(batch(),labels.size());
					for(int i = 0; i < batch(); i++) {
						input.putRow(i,new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)));
						outcomes.put(i,labels.indexOf(windows.get(i).getLabel()),1.0);
					}
					//add left over to cache; need to ensure that only batch() rows are returned
					if(windows.size() > batch()) {
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
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalExamples() {
		return numExamples;
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

	}

	@Override
	public int batch() {
		return batch;
	}

	@Override
	public int cursor() {
		throw new UnsupportedOperationException();

	}

	@Override
	public int numExamples() {
		return numExamples;
	}



}
