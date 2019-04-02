package org.deeplearning4j.iterator.bert;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;


public interface BertSequenceMasker {

    Pair<List<String>,boolean[]> maskSequence(List<String> input, String maskToken, List<String> vocabWords);

}
