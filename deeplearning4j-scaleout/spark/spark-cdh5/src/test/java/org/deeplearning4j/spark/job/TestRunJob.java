package org.deeplearning4j.spark.job;

import static org.junit.Assert.*;


import org.deeplearning4j.spark.impl.multilayer.Worker;
import org.junit.Test;

public class TestRunJob {

	@Test
	public void test() {
		
		String[] args = new String[4];
		args[0] = "src/test/resources/data/svmLight/iris_svmLight_0.txt";
		args[1] = "1";
		args[2] = "0";
		args[3] = "2";
		
		
		Worker.main( args );
		
	}

}
