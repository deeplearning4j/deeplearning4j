package org.deeplearning4j.word2vec.similarity;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.util.SetUtils;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.loader.Word2VecLoader;
import org.deeplearning4j.word2vec.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Word2VecSimilarityTester {

	private static Logger log = LoggerFactory.getLogger(Word2VecSimilarityTester.class);

	public static void main(String[] args) throws Exception {
		Word2Vec vec = Word2VecLoader.loadModel(new File(args[0]));


		File trainingDir = new File(args[1]);

		File[] dirs = trainingDir.listFiles();
		List<File> files = new ArrayList<File>();

		for(File dir : dirs) {
			File[] articles = dir.listFiles();
			for(File f : articles)
				if(f.isFile())
					files.add(f);


		}
		Set<File> alreadyProcessed = new HashSet<File>();
		for(File f1 : files) {
			for(File f2 : files) {
				if(f1.equals(f2))
					continue;
				alreadyProcessed.add(f1);
				String a1 = FileUtils.readFileToString(f1);
				String a2 = FileUtils.readFileToString(f2);
				WordMetaData d1 = new WordMetaData(vec,a1);
				WordMetaData d2 = new WordMetaData(vec,a2);
				d1.calc();
				d2.calc();

				Set<String> vocab = SetUtils.union(d1.getWordCounts().keySet(), d2.getWordCounts().keySet());
				Set<String> remove = new HashSet<String>();
				for(String word : vocab) {
					if(Util.matchesAnyStopWord(vec.getStopWords(),word))
						remove.add(word);
				}
				vocab.removeAll(remove);
				Set<String> inter = SetUtils.intersection(d1.getWordCounts().keySet(), d2.getWordCounts().keySet());
				inter.removeAll(remove);

				List<String> wordList = new ArrayList<>(vocab);

				INDArray a1Matrix = NDArrays.create(wordList.size(), vec.getLayerSize());
				INDArray a2Matrix = NDArrays.create(wordList.size(), vec.getLayerSize());

				for(int i = 0; i < wordList.size(); i++) {
					if(d1.getWordCounts().getCount(wordList.get(i)) > 0) {
						a1Matrix.putRow(i,vec.getWordVectorMatrix(wordList.get(i)));
					}
					else 
						a1Matrix.putRow(i, NDArrays.zeros(vec.getLayerSize()));

					if(d2.getWordCounts().getCount(wordList.get(i)) > 0) {
						a2Matrix.putRow(i,vec.getWordVectorMatrix(wordList.get(i)));

					}
					else 
						a2Matrix.putRow(i, NDArrays.zeros(vec.getLayerSize()));



				}

				double wordSim = (double) inter.size() / (double) wordList.size();
				double finalScore  = Transforms.cosineSim(a1Matrix, a2Matrix) * wordSim;

				if(finalScore >= 0.035) {
					log.info(f1.getName() + " is similar to " + f2.getName() + " with score " + finalScore);
					
				}
			}
		}


	}

}
