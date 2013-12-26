package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.viterbi;

import java.util.*;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.CounterMap;
import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;

public class CounterUtil {

	public static DoubleMatrix convert(CounterMap<Double,Double> counter) {
		DoubleMatrix ret = new DoubleMatrix(counter.size(),counter.keySet().size());
		Iterator<Pair<Double,Double>> iter = counter.getPairIterator();
		
		for(Pair<Double,Double> next = iter.next(); iter.hasNext(); next = iter.next()) {
			double first = next.getFirst();
			double second = next.getSecond();
			int firstInt = (int) first;
			int secondInt = (int) second;
			ret.put(firstInt, secondInt,counter.getCount(next.getFirst(),next.getSecond()));
		}
		
		return ret;
	}

}
