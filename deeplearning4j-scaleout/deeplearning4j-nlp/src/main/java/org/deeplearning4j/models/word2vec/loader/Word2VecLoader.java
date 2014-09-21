package org.deeplearning4j.models.word2vec.loader;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.Index;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads word 2 vec models
 * @author Adam Gibson
 */
public class Word2VecLoader {

    private static final int MAX_SIZE = 50;




    /**
     * Loads the google model
     * @param path the path to the google model
     * @return the loaded model
     * @throws IOException
     */
	public static Word2Vec loadGoogleBinary(String path) throws IOException {
        DataInputStream dis = null;
        BufferedInputStream bis = null;
        double len = 0;
        float vector = 0;
        int words,size = 0;
        InMemoryLookupCache cache;

        try {
            bis = new BufferedInputStream(new FileInputStream(path));
            dis = new DataInputStream(bis);
            words = Integer.parseInt(readString(dis));
            size = Integer.parseInt(readString(dis));

            cache = new InMemoryLookupCache(size,words);

            String word;
            float[] vectors = null;
            for (int i = 0; i < words; i++) {
                word = readString(dis);
                vectors = new float[size];
                len = 0;
                for (int j = 0; j < size; j++) {
                    vector = readFloat(dis);
                    len += vector * vector;
                    vectors[j] = vector;
                }
                len = Math.sqrt(len);

                for (int j = 0; j < size; j++)
                    vectors[j] /= len;


                INDArray row = Nd4j.create(vectors);
                cache.addWordToIndex(cache.numWords(),word);
                cache.putVector(word,row);

                dis.read();
            }
        }
        finally {
            bis.close();
            dis.close();
        }


        Word2Vec ret = new Word2Vec();
        ret.setCache(cache);
        ret.setLayerSize(size);
        return ret;

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
