package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well1024a;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RandomTest {

	private long seed = 123;
	private int iterations = 5;
	private RandomGenerator gen;
	private static Logger log = LoggerFactory.getLogger(RandomTest.class);
	
	@Test
	public void testRandom() {
		gen = new org.apache.commons.math3.random.MersenneTwister(seed);
		
		List<Integer> test1 = testRand();
		gen = new org.apache.commons.math3.random.MersenneTwister(seed);

		List<Integer> test2 = testRand();
		
		log.info(String.valueOf(test1.equals(test2)));
	}
	
	private List<Integer> testRand() {
		List<Integer> ret = new ArrayList<Integer>();
		for(int i = 0; i < iterations; i++)  {
			ret.add(gen.nextInt());
		}
		return ret;
	}
	
	

}
