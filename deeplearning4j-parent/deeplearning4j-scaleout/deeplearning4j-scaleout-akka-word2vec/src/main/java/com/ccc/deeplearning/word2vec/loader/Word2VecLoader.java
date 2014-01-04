package com.ccc.deeplearning.word2vec.loader;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.word2vec.VocabWord;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.viterbi.Index;

public class Word2VecLoader {

	private static Logger log = LoggerFactory.getLogger(Word2VecLoader.class);

	public static Word2Vec loadModel(File file) throws Exception {
		log.info("Loading model from " + file.getAbsolutePath());
		Word2Vec ret = new Word2Vec();
		try(DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
			int vocabSize = dis.readInt();
			int layerSize = dis.readInt();
			ret.setLayerSize(layerSize);
			for(int i = 0; i < vocabSize; i++) {
				String word = dis.readUTF();
				ret.getWordIndex().add(word);
				ret.getVocab().put(word,new VocabWord().read(dis, layerSize));

			}
			ret.setSyn0(DoubleMatrix.zeros(vocabSize, layerSize));
			ret.setSyn1(DoubleMatrix.zeros(vocabSize, layerSize));

			ret.getSyn0().in(dis);
			ret.getSyn1().in(dis);

			dis.close();

		}
		catch(IOException e) {
			log.error("Unable to read file for loading model",e);
		}
		return ret;
	}



	public static Word2Vec loadTextModel(File file) throws IOException {
		@SuppressWarnings("unchecked")
		List<String> list = FileUtils.readLines(file);
		Word2Vec vec = new Word2Vec();
		vec.buildVocab(list);
		for(String sentence : list)
			vec.addSentence(sentence);
		vec.train();
		return vec;

	}

	public static Word2Vec loadBinary(File file) throws IOException {
		Word2Vec vec = new Word2Vec();
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(
				file)))) {
			int words = dis.readInt();
			int size = dis.readInt();
			vec.setLayerSize(size);
			double len = 0;
			double vector = 0;
			vec.setSyn0(DoubleMatrix.zeros(words, size));
			vec.setSyn1(DoubleMatrix.zeros(words,size));

			String key = null;
			double[] value = null;
			for (int i = 0; i < words; i++) {
				key = dis.readUTF();
				value = new double[size];
				for (int j = 0; j < size; j++) {
					vector = dis.readDouble();
					len += vector * vector;
					value[j] = vector;
				}

				len = Math.sqrt(len);

				for (int j = 0; j < size; j++) 
					value[j] /= len;
				vec.getWordIndex().add(key);
				vec.getSyn0().putRow(i,new DoubleMatrix(value));

			}

		}
		return vec;
	}

	private static Word2Vec loadGoogleVocab(Word2Vec vec,String path) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		String temp = null;
		vec.setWordIndex(new Index());
		vec.getVocab().clear();
		while((temp = reader.readLine()) != null) {
			String[] split = temp.split(" ");
			if(split[0].equals("</s>"))
				continue;

			int freq = Integer.parseInt(split[1]);
			VocabWord realWord = new VocabWord(freq,vec.getLayerSize());
			realWord.setIndex(vec.getVocab().size());
			vec.getVocab().put(split[0], realWord);
			vec.getWordIndex().add(split[0]);
		}
		reader.close();
		return vec;
	}


	public static Word2Vec loadGoogleText(String path,String vocabPath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		String temp = null;
		boolean first = true;
		Integer vectorSize = null;
		Integer rows = null;
		int currRow = 0;
		Word2Vec ret = new Word2Vec();
		while((temp = reader.readLine()) != null) {
			if(first) {
				String[] split = temp.split(" ");
				rows = Integer.parseInt(split[0]);
				vectorSize = Integer.parseInt(split[1]);
				ret.setLayerSize(vectorSize);
				ret.setSyn0(new DoubleMatrix(rows - 1,vectorSize));
				first = false;
			}

			else {
				StringTokenizer tokenizer = new StringTokenizer(temp);
				double[] vec = new double[ret.getLayerSize()];
				int count = 0;
				String word = tokenizer.nextToken();
				if(word.equals("</s>"))
					continue;

				while(tokenizer.hasMoreTokens()) {
					vec[count++] = Double.parseDouble(tokenizer.nextToken());
				}
				ret.getSyn0().putRow(currRow, new DoubleMatrix(vec));
				currRow++;

			}
		}
		reader.close();

		return loadGoogleVocab(ret,vocabPath);

	}




}
