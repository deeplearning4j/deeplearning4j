package com.ccc.deeplearning.word2vec.viterbi;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.util.MatrixUtil;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.util.Window;
import com.ccc.deeplearning.word2vec.util.WindowConverter;
import com.ccc.deeplearning.word2vec.util.Windows;

public class ViterbiTest {

	@SuppressWarnings("unchecked")
	@Test
	public void testDecoding() throws Exception {
		Index labelIndex = new Index();
		labelIndex.add("NONE");
		labelIndex.add("ADDRESS");
		Index featureIndex = ViterbiUtil.featureIndexFromLabelIndex(labelIndex);
		List<String> lines = IOUtils.readLines(new ClassPathResource("/deeplearning/ADDRESS-small.split").getInputStream());
		CounterMap<Integer,Integer> transitions = new CounterMap<Integer,Integer>();
		
		Word2Vec vec = Word2VecLoader.loadModel(new ClassPathResource("/word2vec-address.bin").getFile());
		
		for(String line : lines) {
			if(line.isEmpty()) 
				continue;
			List<Window> windows = Windows.windows(line);
		
			for(int i = 1; i < windows.size(); i++) {
				String firstLabel = windows.get(i - 1).getLabel();
				String secondLabel = windows.get(i).getLabel();
				int first = labelIndex.indexOf(firstLabel);
				int second = labelIndex.indexOf(secondLabel);
				
				
				transitions.incrementCount(first,second,1.0);
			}
			
		}
		
		DoubleMatrix transitionWeights = CounterUtil.convert(transitions);
		Viterbi viterbi = new Viterbi(labelIndex,featureIndex,transitionWeights);
		for(String line : lines) {
			if(line.isEmpty()) 
				continue;
			List<Window> windows = Windows.windows(line);
			List<DoubleMatrix> classified = new ArrayList<DoubleMatrix>();
			for(Window w : windows) {
				classified.add(MatrixUtil.toOutcomeVector(labelIndex.indexOf(w.getLabel()), labelIndex.size()));
			}
			DoubleMatrix classifiedM = new DoubleMatrix(classified.size(), classified.get(0).columns);
			for(int i = 0; i < classified.size(); i++) {
				classifiedM.putRow(i,classified.get(i));
			}
			
			
			List<String> labels = new ArrayList<String>();
			labels.add("NONE");
			labels.add("ADDRESS");
			
			viterbi.decode(classifiedM,labels,windows);
			
		}
	}

}
