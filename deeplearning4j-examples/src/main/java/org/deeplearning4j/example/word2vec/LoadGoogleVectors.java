package org.deeplearning4j.example.word2vec;

import org.deeplearning4j.util.SerializationUtils;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.loader.Word2VecLoader;

import java.io.File;

/**
 * Created by agibsonccc on 5/25/14.
 */
public class LoadGoogleVectors {

    public static void main(String[] args) throws Exception {
        Word2Vec vec = Word2VecLoader.loadGoogleModel(args[0]);
        SerializationUtils.saveObject(vec,new File(args[1]));
    }


}
