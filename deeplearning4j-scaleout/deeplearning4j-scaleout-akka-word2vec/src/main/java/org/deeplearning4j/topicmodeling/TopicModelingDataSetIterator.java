package org.deeplearning4j.topicmodeling;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.iterativereduce.actor.core.SerializableFileIter;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.util.FeatureUtil;
import org.deeplearning4j.util.Index;


public class TopicModelingDataSetIterator implements DataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5137504154841977611L;
	private List<String> labels;
	private Index vocab = new Index();
	private VocabCreator vocabCreator;
	private SerializableFileIter files;
	private int batchSize = 1;

	/**
	 * 
	 * @param rootDir the root directory to getFromOrigin data from
	 * @param numWords the number of words to applyTransformToDestination as vocab
	 * @param batchSize the batch size to train on
	 */
	@SuppressWarnings("unchecked")
	public TopicModelingDataSetIterator(File rootDir,int numWords,int batchSize) {
		File[] subs = rootDir.listFiles();
		if(subs == null)
			throw new IllegalArgumentException("Root directory has no files");
		
		
		labels = new ArrayList<String>(subs.length);
		for(File f : subs) {
			labels.add(f.getName());
		}
		

		this.vocabCreator = new VocabCreator(rootDir);
		vocab = vocabCreator.createVocab(numWords);
		Iterator<File> iter = FileUtils.iterateFiles(rootDir, null, true);
		List<File> list = new ArrayList<File>();
		while(iter.hasNext()) 
			list.add(iter.next());

		
		Collections.shuffle(list);
		files = new SerializableFileIter(list);


	}

	
	/**
	 * This assumes a previously persisted vocab creator was used.
	 * @param rootDir the root directory to train on
	 * @param batchSize the batch size, or number of rows
	 * in the data applyTransformToDestination
	 * @param vocabCreator the vocab creator 
	 */
	@SuppressWarnings("unchecked")
	public TopicModelingDataSetIterator(File rootDir,int batchSize,VocabCreator vocabCreator) {
		File[] subs = rootDir.listFiles();
		if(subs == null)
			throw new IllegalArgumentException("Root directory has no files");
		
		
		labels = new ArrayList<String>(subs.length);
		this.batchSize = batchSize;
		for(File f : subs) {
			labels.add(f.getName());
		}
		

		this.vocabCreator = vocabCreator;
		vocab = vocabCreator.getCurrVocab();
		Iterator<File> iter = FileUtils.iterateFiles(rootDir, null, true);
		List<File> list = new ArrayList<File>();
		while(iter.hasNext()) 
			list.add(iter.next());

		
		Collections.shuffle(list);
		files = new SerializableFileIter(list);


	}

	@Override
	public  boolean hasNext() {
		return files.hasNext();


	}

	@Override
	public DataSet next() {
		int curr = 0;
		if(!hasNext())
			throw new IllegalStateException("Unable to getFromOrigin next; no more data found");

		List<DataSet> d = new ArrayList<>();
		while(files.hasNext() && curr  < batchSize) {
			File next = files.next();
			INDArray input = vocabCreator.getScoreMatrix(next);
            int comp = (int) input.sum(Integer.MAX_VALUE).element();
			if(comp != 0) {
				String label = next.getParentFile().getName();
				INDArray y = FeatureUtil.toOutcomeVector(labels.indexOf(label), labels.size());
				d.add(new DataSet(input,y));
				curr++;
			}


		}

		return DataSet.merge(d);

	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalExamples() {
		throw new UnsupportedOperationException();

	}

	@Override
	public int inputColumns() {
		return vocab.size();
	}

	@Override
	public int totalOutcomes() {
		return this.labels.size();
	}

	@Override
	public void reset() {
		files.setCurr(0);
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		throw new UnsupportedOperationException();

	}

	@Override
	public int numExamples() {
		throw new UnsupportedOperationException();

	}


	@Override
	public DataSet next(int num) {
		int curr = 0;
		if(!hasNext())
			throw new IllegalStateException("Unable to getFromOrigin next; no more data found");

		List<DataSet> d = new ArrayList<>();
		while(files.hasNext() && curr  < num) {
			File next = files.next();
			INDArray input = vocabCreator.getScoreMatrix(next);
            int comp = (int) input.sum(Integer.MAX_VALUE).element();
			if(comp != 0) {
				String label = next.getParentFile().getName();
				INDArray y = FeatureUtil.toOutcomeVector(labels.indexOf(label), labels.size());
				d.add(new DataSet(input,y));
				curr++;
			}


		}

		return DataSet.merge(d);

	}

	





}
