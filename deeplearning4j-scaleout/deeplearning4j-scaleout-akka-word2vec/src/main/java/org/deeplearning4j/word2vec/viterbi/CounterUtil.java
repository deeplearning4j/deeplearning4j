package org.deeplearning4j.word2vec.viterbi;

import java.util.*;

import org.deeplearning4j.berkeley.CounterMap;
import org.deeplearning4j.berkeley.Pair;
import org.jblas.DoubleMatrix;


public class CounterUtil {
	
	public static DoubleMatrix convert(CounterMap<Integer,Integer> counter) {
		DoubleMatrix ret = new DoubleMatrix(counter.size(),counter.keySet().size());
		Iterator<Pair<Integer,Integer>> iter = counter.getPairIterator();
		
		for(Pair<Integer,Integer> next = iter.next(); iter.hasNext(); next = iter.next()) {
			int firstInt = next.getFirst();
			int secondInt = next.getSecond();
			ret.put(firstInt, secondInt,counter.getCount(next.getFirst(),next.getSecond()));
		}
		
		return ret;
	}

}
