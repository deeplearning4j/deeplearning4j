package org.deeplearning4j.word2vec.util;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.deeplearning4j.word2vec.Word2Vec;


public class WordConverter {

	private List<String> sentences = new ArrayList<String>();
	private Word2Vec vec;
	private List<Window> windows;

	public WordConverter(List<String> sentences,Word2Vec vec) {
		this.sentences = sentences;
		this.vec = vec;
	}

	public static INDArray toInputMatrix(List<Window> windows,Word2Vec vec) {
		int columns = vec.getLayerSize() * vec.getWindow();
		int rows = windows.size();
		INDArray ret = Nd4j.create(rows,columns);
		for(int i = 0; i < rows; i++) {
			ret.putRow(i, Nd4j.create(WindowConverter.asExample(windows.get(i), vec)));
		}
		return ret;
	}
	
	
	public INDArray toInputMatrix() {
		List<Window> windows = allWindowsForAllSentences();
		return toInputMatrix(windows,vec);
	}

	

	public static INDArray toLabelMatrix(List<String> labels,List<Window> windows) {
		int columns = labels.size();
		INDArray ret = Nd4j.create(windows.size(),columns);
		for(int i = 0; i < ret.rows(); i++) {
			ret.putRow(i, FeatureUtil.toOutcomeVector(labels.indexOf(windows.get(i).getLabel()), labels.size()));
		}
		return ret;
	}
	
	public INDArray toLabelMatrix(List<String> labels) {
		List<Window> windows = allWindowsForAllSentences();
		return toLabelMatrix(labels,windows);
	}

	private List<Window> allWindowsForAllSentences() {
		if(windows != null)
			return windows;
		windows = new ArrayList<Window>();
		for(String s : sentences)
			if(!s.isEmpty())
				windows.addAll(Windows.windows(s));
		return windows;
	}



}
