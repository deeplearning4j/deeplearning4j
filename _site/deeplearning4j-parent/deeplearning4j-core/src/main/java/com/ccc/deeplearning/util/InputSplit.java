package com.ccc.deeplearning.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Pair;

public class InputSplit {

	public static void splitInputs(DoubleMatrix inputs,DoubleMatrix outcomes,List<Pair<DoubleMatrix,DoubleMatrix>> train,List<Pair<DoubleMatrix,DoubleMatrix>> test,double split) {
		List<DoubleMatrix> inputRows = inputs.rowsAsList();
		List<DoubleMatrix> outcomeRows = outcomes.rowsAsList();
		assert inputRows.size() == outcomeRows.size();
		List<Pair<DoubleMatrix,DoubleMatrix>> list = new ArrayList<>();
		for(int i = 0; i < inputRows.size(); i++) {
			list.add(new Pair<>(inputRows.get(i),outcomeRows.get(i)));
		}

		splitInputs(list,train,test,split);
	}

	public static void splitInputs(List<Pair<DoubleMatrix,DoubleMatrix>> pairs,List<Pair<DoubleMatrix,DoubleMatrix>> train,List<Pair<DoubleMatrix,DoubleMatrix>> test,double split) {
		Random rand = new Random();

		for(Pair<DoubleMatrix,DoubleMatrix> pair : pairs)
			if(rand.nextDouble() <= split) 
				train.add(pair);
			else
				test.add(pair);

			
	}

}
