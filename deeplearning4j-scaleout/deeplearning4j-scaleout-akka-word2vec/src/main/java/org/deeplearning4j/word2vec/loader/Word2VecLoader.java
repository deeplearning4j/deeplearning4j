package org.deeplearning4j.word2vec.loader;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.word2vec.VocabWord;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.util.Index;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Word2VecLoader {

	private static Logger log = LoggerFactory.getLogger(Word2VecLoader.class);
    private static final int MAX_SIZE = 50;

	public static Word2Vec loadModel(File file) throws Exception {
		log.info("Loading model from " + file.getAbsolutePath());
		Word2Vec ret = new Word2Vec();
		ret.load(new BufferedInputStream(new FileInputStream(file)));
		return ret;
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
				ret.setSyn0(NDArrays.create(rows - 1,vectorSize));
				first = false;
			}

			else {
				StringTokenizer tokenizer = new StringTokenizer(temp);
				float[] vec = new float[ret.getLayerSize()];
				int count = 0;
				String word = tokenizer.nextToken();
				if(word.equals("</s>"))
					continue;

				while(tokenizer.hasMoreTokens()) {
					vec[count++] = Float.parseFloat(tokenizer.nextToken());
				}
				ret.getSyn0().putRow(currRow, NDArrays.create(vec));
				currRow++;

			}
		}
		reader.close();

		return loadGoogleVocab(ret,vocabPath);

	}


    /**
     * Read a string from a data input stream
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     * @param dis
     * @return
     * @throws IOException
     */
    private static String readString(DataInputStream dis) throws IOException {
        byte[] bytes = new byte[MAX_SIZE];
        byte b = dis.readByte();
        int i = -1;
        StringBuilder sb = new StringBuilder();
        while (b != 32 && b != 10) {
            i++;
            bytes[i] = b;
            b = dis.readByte();
            if (i == 49) {
                sb.append(new String(bytes));
                i = -1;
                bytes = new byte[MAX_SIZE];
            }
        }
        sb.append(new String(bytes, 0, i + 1));
        return sb.toString();
    }


    /**
     * Loads the google binary model
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     * @param path path to model
     *
     * @throws IOException
     */
    public static Word2Vec loadGoogleModel(String path) throws IOException {
        DataInputStream dis = null;
        BufferedInputStream bis = null;
        double len = 0;
        float vector = 0;
        Word2Vec ret = new Word2Vec();
        Index wordIndex = new Index();
        INDArray wordVectors = null;
        try {
            bis = new BufferedInputStream(path.endsWith(".gz") ? new GZIPInputStream(new FileInputStream(path)) : new FileInputStream(path));
            dis = new DataInputStream(bis);
            Map<String,INDArray> wordMap = new HashMap<>();
            //number of words
            int words = Integer.parseInt(readString(dis));
            //word vector size
            int size = Integer.parseInt(readString(dis));
            wordVectors = NDArrays.create(words,size);
            String word;
            float[] vectors = null;
            for (int i = 0; i < words; i++) {
                word = readString(dis);
                log.info("Loaded " + word);
                vectors = new float[size];
                len = 0;
                for (int j = 0; j < size; j++) {
                    vector = readFloat(dis);
                    len += vector * vector;
                    vectors[j] =  vector;
                }
                len = Math.sqrt(len);

                for (int j = 0; j < size; j++) {
                    vectors[j] /= len;
                }
                wordIndex.add(word);
                wordVectors.putRow(i, NDArrays.create(vectors));
            }
        } finally {
            bis.close();
            dis.close();
        }

        ret.setWordIndex(wordIndex);
        ret.setSyn0(wordVectors);

        return ret;
    }

    /**
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     * @param is
     * @return
     * @throws IOException
     */
    public static float readFloat(InputStream is) throws IOException {
        byte[] bytes = new byte[4];
        is.read(bytes);
        return getFloat(bytes);
    }

    /**
     * Read float from byte array, credit to:
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     *
     * @param b
     * @return
     */
    public static float getFloat(byte[] b) {
        int accum = 0;
        accum = accum | (b[0] & 0xff) << 0;
        accum = accum | (b[1] & 0xff) << 8;
        accum = accum | (b[2] & 0xff) << 16;
        accum = accum | (b[3] & 0xff) << 24;
        return Float.intBitsToFloat(accum);
    }





}
