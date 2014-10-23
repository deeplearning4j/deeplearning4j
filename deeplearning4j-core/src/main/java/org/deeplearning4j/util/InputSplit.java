package org.deeplearning4j.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;


public class InputSplit {

	public static void splitInputs(INDArray inputs,INDArray outcomes,List<Pair<INDArray,INDArray>> train,List<Pair<INDArray,INDArray>> test,double split) {
		List<Pair<INDArray,INDArray>> list = new ArrayList<>();
		for(int i = 0; i < inputs.rows(); i++) {
			list.add(new Pair<>(inputs.getRow(i),outcomes.getRow(i)));
		}

		splitInputs(list,train,test,split);
	}

	public static void splitInputs(List<Pair<INDArray,INDArray>> pairs,List<Pair<INDArray,INDArray>> train,List<Pair<INDArray,INDArray>> test,double split) {
		Random rand = new Random();

		for(Pair<INDArray,INDArray> pair : pairs)
			if(rand.nextDouble() <= split) 
				train.add(pair);
			else
				test.add(pair);

			
	}

}
