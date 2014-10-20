package org.deeplearning4j.models.word2vec;

import java.io.InputStream;
import java.io.Serializable;

/**
 * Created by agibsonccc on 10/19/14.
 */
public interface InputStreamCreator extends Serializable {



    public InputStream create();
}
