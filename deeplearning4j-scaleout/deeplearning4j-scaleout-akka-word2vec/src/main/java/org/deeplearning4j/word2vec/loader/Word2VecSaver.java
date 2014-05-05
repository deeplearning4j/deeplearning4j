package org.deeplearning4j.word2vec.loader;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;

import org.deeplearning4j.util.FileOperations;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.util.Index;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;


public class Word2VecSaver {


	private static Logger log = LoggerFactory.getLogger(Word2VecSaver.class);
	
	
	public void saveAsCsv(File path,Word2Vec vec) {
		Index wordIndex = vec.getWordIndex();
		DoubleMatrix syn0 = vec.getSyn0();
		
		//change each row to a column: this allows each column to be a word
		DoubleMatrix toSave = MatrixUtil.normalizeByRowSums(syn0.transpose());
		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		for(int i = 0; i < wordIndex.size(); i++) {
			atts.add(new Attribute(wordIndex.get(i).toString()));
		}

		Instances instances = new Instances("",atts, toSave.rows);

		OutputStream os = FileOperations.createAppendingOutputStream(path);
		try {



			for(int i = 0; i < toSave.rows; i++)  {
				DoubleMatrix row = toSave.getRow(i);
				instances.add(new DenseInstance(1.0, row.toArray()));
			}

			os.write(instances.toString().getBytes());
			os.flush();
			os.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}


	}
	
	/**
	 * The format for saving the word2vec model is as follows.
	 * 
	 * @param file
	 * @throws IOException
	 */
	public static void saveModel(Word2Vec vec,File file) throws IOException {

		if(file.exists())
			file.delete();

		try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
				new FileOutputStream(file)))) {
			dataOutputStream.writeInt(vec.getVocab().size());
			dataOutputStream.writeInt(vec.getLayerSize());

			for(int i = 0; i < vec.getVocab().size(); i++) {
				String word = vec.getWordIndex().get(i).toString();
				dataOutputStream.writeUTF(word);
				vec.getVocab().get(word).write(dataOutputStream);

			}

			vec.getSyn0().out(dataOutputStream);
			vec.getSyn1().out(dataOutputStream);



			dataOutputStream.flush();
			dataOutputStream.close();

		} catch (IOException e) {
			log.error("Unable to save model",e);
		}

	}
}
