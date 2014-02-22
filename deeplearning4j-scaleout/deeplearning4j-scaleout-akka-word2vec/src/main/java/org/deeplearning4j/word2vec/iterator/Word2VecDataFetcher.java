package org.deeplearning4j.word2vec.iterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetFetcher;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.util.Window;
import org.deeplearning4j.word2vec.util.WindowConverter;
import org.deeplearning4j.word2vec.util.Windows;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@SuppressWarnings("unchecked")
public class Word2VecDataFetcher implements DataSetFetcher {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3245955804749769475L;
	private transient Iterator<File> files;
	private Word2Vec vec;
	private static Pattern begin = Pattern.compile("<[A-Z]+>");
	private static Pattern end = Pattern.compile("</[A-Z]+>");
	private List<String> labels = new ArrayList<String>();
	private int batch;
	private List<Window> cache = new ArrayList<Window>();
	private static Logger log = LoggerFactory.getLogger(Word2VecDataFetcher.class);
	private int totalExamples;
	private String path;
	
	public Word2VecDataFetcher(String path,Word2Vec vec,List<String> labels) {
		if(vec == null || labels == null || labels.isEmpty())
			throw new IllegalArgumentException("Unable to initialize due to missing argument or empty label set");
		this.vec = vec;
		this.labels = labels;
		this.path = path;
	}



	private DataSet fromCache() {
		DoubleMatrix outcomes = null;
		DoubleMatrix input = null;
		input = new DoubleMatrix(batch,vec.getSyn1().columns * vec.getWindow());
		outcomes = new DoubleMatrix(batch,labels.size());
		for(int i = 0; i < batch; i++) {
			input.putRow(i,new DoubleMatrix(WindowConverter.asExample(cache.get(i), vec)));
			int idx = labels.indexOf(cache.get(i).getLabel());
			if(idx < 0)
				idx = 0;
			outcomes.putRow(i,MatrixUtil.toOutcomeVector(idx, labels.size()));
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
				if(windows.isEmpty() && lines.hasNext())
		              continue;
				
				if(windows.size() < batch) {
					input = new DoubleMatrix(windows.size(),vec.getSyn1().columns * vec.getWindow());
					outcomes = new DoubleMatrix(batch,labels.size());
					for(int i = 0; i < windows.size(); i++) {
						input.putRow(i,new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)));
						int idx = labels.indexOf(windows.get(i).getLabel());
						if(idx < 0)
							idx = 0;
						DoubleMatrix outcomeRow = MatrixUtil.toOutcomeVector(idx, labels.size());
						outcomes.putRow(i,outcomeRow);
					}
					return new DataSet(input,outcomes);


				}
				else {
					input = new DoubleMatrix(batch,vec.getSyn1().columns * vec.getWindow());
					outcomes = new DoubleMatrix(batch,labels.size());
					for(int i = 0; i < batch; i++) {
						input.putRow(i,new DoubleMatrix(WindowConverter.asExample(windows.get(i), vec)));
						int idx = labels.indexOf(windows.get(i).getLabel());
						if(idx < 0)
							idx = 0;
						DoubleMatrix outcomeRow = MatrixUtil.toOutcomeVector(idx, labels.size());
						outcomes.putRow(i,outcomeRow);
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
		return vec.getLayerSize() * vec.getWindow();
	}

	@Override
	public int totalOutcomes() {
		return labels.size();

	}

	@Override
	public void reset() {
		files = FileUtils.iterateFiles(new File(path), null, true);
		cache.clear();

	}



	@Override
	public int cursor() {
		return 0;

	}



	@Override
	public boolean hasMore() {
		return files.hasNext() || !cache.isEmpty();
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
