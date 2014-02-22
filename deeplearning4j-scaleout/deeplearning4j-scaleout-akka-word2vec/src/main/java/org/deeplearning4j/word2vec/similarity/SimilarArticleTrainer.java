package org.deeplearning4j.word2vec.similarity;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.loader.Word2VecLoader;
import org.deeplearning4j.word2vec.similarity.WordMetaData;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class SimilarArticleTrainer {

	private static Logger log = LoggerFactory.getLogger(SimilarArticleTrainer.class);


	public static void main(String[] args) throws Exception {
		Word2Vec vec = Word2VecLoader.loadModel(new File(args[0]));


		File trainingDir = new File(args[1]);

		File[] dirs = trainingDir.listFiles();
		List<DoubleMatrix> individualMatrices = new ArrayList<DoubleMatrix>();
		List<DataSet> examples = new ArrayList<DataSet>();
		for(File dir : dirs) {
			if(!dir.getName().equals("test")) {
				File[] articles = dir.listFiles();
				String a1 = FileUtils.readFileToString(articles[0]);
				String a2 = FileUtils.readFileToString(articles[1]);
				WordMetaData d1 = new WordMetaData(vec,a1);
				WordMetaData d2 = new WordMetaData(vec,a2);
				DoubleMatrix positive = new DoubleMatrix(new double[]{1.0,0.0});

				DoubleMatrix firstPositive = matrixFor(vec,d1,vec.getVocab().keySet());
				DoubleMatrix secondPositive = matrixFor(vec,d2,vec.getVocab().keySet());
				individualMatrices.add(firstPositive);
				individualMatrices.add(secondPositive);
				List<DataSet> positiveExamples = getExamples(firstPositive,secondPositive,positive);
				examples.addAll(positiveExamples);

			}
		}

		Random r = new Random();
		CDBN dbn  = new CDBN.Builder()
		.numberOfInputs(examples.get(0).getFirst().columns)
		.numberOfOutPuts(2)
		.hiddenLayerSizes(new int[]{500,500,2000}).build();

		for(int i = 0; i < individualMatrices.size(); i++) {
			DoubleMatrix negative = new DoubleMatrix(new double[]{1.0,0.0});
			int negativeExample = r.nextInt(individualMatrices.size());
			while(negativeExample == i || (Math.abs(negativeExample - i)) == 1)
				negativeExample = r.nextInt(individualMatrices.size());

			List<DataSet> negativeExamples = getExamples(individualMatrices.get(0),individualMatrices.get(negativeExample),negative);
			examples.addAll(negativeExamples);
		}

		DataSet d = DataSet.merge(examples);



		log.info("Training on " + d.numExamples() + " with " + examples.get(0).getFirst().columns + " inputs");

		List<List<DataSet>> batches = d.batchBy(10);

		d = null;
		examples.clear();
		individualMatrices.clear();
		
		for(List<DataSet> list : batches) {
			DataSet train = DataSet.merge(list);
			dbn.pretrain(train.getFirst(),3, 0.1, 5);
			dbn.finetune(train.getSecond(),0.1, 5);
		}

		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(new File(args[2])));
		dbn.write(bos);
		bos.flush();
		bos.close();
	}

	private static DoubleMatrix matrixFor(Word2Vec vec,WordMetaData d,Collection<String> vocab) {
		List<String> validWords = new ArrayList<String>();
		for(String word : vocab) {
			validWords.add(word);
		}

		DoubleMatrix m1 = new DoubleMatrix(validWords.size(),vec.getLayerSize());
		for(int i = 0; i < validWords.size(); i++) {
			if(d.getWordCounts().getCount(validWords.get(i)) > 0)
				m1.putRow(i, d.getVectorForWord(validWords.get(i)));
			else
				m1.putRow(i,DoubleMatrix.zeros(vec.getLayerSize()));
		}

		return m1;
	}


	private static List<DataSet> getExamples(DoubleMatrix v1, DoubleMatrix v2,DoubleMatrix outcome) {
		List<DataSet>  ret = new ArrayList<>();
		ret.add(new DataSet(combine(v1,v2),outcome.dup()));
		ret.add(new DataSet(combine(v2,v1),outcome.dup()));
		return ret;
	}

	private static DoubleMatrix combine(DoubleMatrix first,DoubleMatrix second) {
		DoubleMatrix ret = new DoubleMatrix(1,first.length + second.length);
		for(int i = 0; i < first.length; i++) {
			ret.put(i,first.get(i));
		}

		for(int i = 0; i < second.length; i++) {
			int adjusted = i + first.length;
			ret.put(adjusted,second.get(i));
		}
		return ret;
	}



}
