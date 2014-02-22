package org.deeplearning4j.word2vec.viterbi;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.word2vec.util.Window;
import org.deeplearning4j.word2vec.util.Windows;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@SuppressWarnings("unchecked")
public class ViterbiTrainer {

	private static Logger log = LoggerFactory.getLogger(ViterbiTrainer.class);
	
	public static void main(String[] args) throws IOException {
		String[] labels = args[0].split(",");
		String dir = args[1];
		String output = args[2];
		Iterator<File> files = FileUtils.iterateFiles(new File(dir), null, true);
		Index labelIndex = new Index();
		for(int i = 0; i < labels.length; i++)
			labelIndex.add(labels[i]);
		Index featureIndex = ViterbiUtil.featureIndexFromLabelIndex(labelIndex);
		CounterMap<Integer,Integer> transitions = new CounterMap<Integer,Integer>();

		while(files.hasNext()) {
			File file = files.next();
			log.info("Loading " + file.getAbsolutePath());

			List<String> lines = FileUtils.readLines(file);
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
		}


		DoubleMatrix transitionWeights = CounterUtil.convert(transitions);
		Viterbi viterbi = new Viterbi(labelIndex,featureIndex,transitionWeights);
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(output));
		viterbi.write(bos);
		bos.flush();
		bos.close();
		log.info("Saved to " + output);
	}

}
