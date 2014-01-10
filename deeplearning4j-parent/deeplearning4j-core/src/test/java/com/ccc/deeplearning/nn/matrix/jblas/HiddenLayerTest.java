package com.ccc.deeplearning.nn.matrix.jblas;

import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.nn.HiddenLayer;

public class HiddenLayerTest {

	private static Logger log = LoggerFactory.getLogger(HiddenLayerTest.class);
	
	@Test
	public void testSimple() {
		DoubleMatrix input = new DoubleMatrix(new double[][] 
		{{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
        ,{1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
        ,{1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0}
        ,{1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
        ,{0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}
        ,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1}
        ,{0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1}
        ,{0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1}
        ,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1}
        ,{0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0}});
		
		HiddenLayer layer = new HiddenLayer(20, 2, null, null, null, input);
		log.info(layer.sample_h_given_v().toString());
		log.info(layer.outputMatrix().toString());
		
		
	}

}
