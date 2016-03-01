package org.nd4j.linalg.ops.impl.transform;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

@RunWith(Parameterized.class)
public class TestDerivatives extends BaseNd4jTest {
	
	public static final double REL_ERROR_TOLERANCE = 1e-3;


    public TestDerivatives(Nd4jBackend backend) {
        super(backend);
    }


    static {
		Nd4j.dtype = DataBuffer.Type.DOUBLE;
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);
	}
	
	@Test
	public void testHardTanhDerivative(){
		assertTrue( Nd4j.getOpFactory().createTransform("hardtanh", Nd4j.ones(1)).derivative() instanceof HardTanhDerivative);
		
		//HardTanh:
		//f(x) = 1 if x > 1
		//f(x) = -1 if x < -1
		//f(x) = x otherwise
		//This is piecewise differentiable.
		//f'(x) = 0 if |x|>1
		//f'(x) = 1 otherwise
		//Note for x= +/- 1, HardTanh is not differentiable. Choose f'(+/- 1) = 1
		
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i=0; i < 100; i++) {
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			expOut[i] = (Math.abs(x) <= 1.0 ? 1 : 0);
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("hardtanh", z).derivative());
		
		for( int i = 0; i < 100; i++ ){
			assertEquals(expOut[i],zPrime.getDouble(i),1e-1);
		}
	}
	

	@Test
	public void testRectifiedLinearDerivative(){
		assertTrue( Nd4j.getOpFactory().createTransform("relu", Nd4j.ones(1)).derivative() instanceof Step );
		
		//ReLU:
		//f(x) = max(0,x)
		//Piecewise differentiable; choose f'(0) = 0
		//f'(x) = 1 if x > 0
		//f'(x) = 0 if x <= 0
		
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i = 0; i<100; i++ ){
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			expOut[i] = (x  > 0 ? 1 : 0);
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("relu", z).derivative());
		
		for( int i = 0; i < 100; i++ ){
			assertTrue(expOut[i] == zPrime.getDouble(i));
		}
	}
	
	@Test
	public void testSigmoidDerivative(){
		assertTrue( Nd4j.getOpFactory().createTransform("sigmoid", Nd4j.ones(1)).derivative() instanceof SigmoidDerivative );
		
		//Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
		//s(x) = 1 / (exp(-x) + 1)
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i = 0; i < 100; i++ ){
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			double sigmoid = 1.0 / (FastMath.exp(-x)+1);
			expOut[i] = sigmoid * (1-sigmoid);
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", z).derivative());
		
		for( int i = 0; i < 100; i++) {
			double relError = Math.abs(expOut[i]-zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
			assertTrue(relError < REL_ERROR_TOLERANCE);
		}
	}
	
	@Test
	public void testSoftMaxDerivative(){
		assertTrue( Nd4j.getOpFactory().createTransform("softmax", Nd4j.ones(1)).derivative() instanceof SoftMaxDerivative );
		
		Random r = new Random(12345L);
		
		INDArray z = Nd4j.zeros(20,5);
		double[][] in = new double[20][5];
		double[][] softmax = new double[20][5];
		double[][] expOut = new double[20][5];
		for( int i = 0; i < 20; i++) {
			double rowSumExp = 0.0;
			for( int j = 0; j < 5; j++) {
				in[i][j] = 10*r.nextDouble();
				z.putScalar(new int[]{i,j}, in[i][j]);
				rowSumExp += FastMath.exp(in[i][j]);
			}
			for(int j = 0; j < 5; j++){
				softmax[i][j] = FastMath.exp(in[i][j]) / rowSumExp;
				expOut[i][j] = softmax[i][j] * (1.0 - softmax[i][j]);
			}
		}
		
		INDArray sm = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", z.dup()), 1);
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", z).derivative());
		System.out.println(Arrays.toString(sm.data().asDouble()));
		System.out.println(Arrays.toString(zPrime.data().asDouble()));
		assertNotEquals(sm,zPrime);
		
		for( int i=0; i < 20; i++) {
			for( int j=0; j < 5; j++) {
				double relError = Math.abs(expOut[i][j]-zPrime.getDouble(i,j)) / (Math.abs(expOut[i][j]) + Math.abs(zPrime.getDouble(i,j)));
				assertTrue(relError < REL_ERROR_TOLERANCE);
			}
		}
	}
	

	@Test
	public void testSoftPlusDerivative(){
		//Derivative of softplus in sigmoid
		assertTrue( Nd4j.getOpFactory().createTransform("softplus", Nd4j.ones(1)).derivative() instanceof Sigmoid );
		
		//s(x) = 1 / (exp(-x) + 1)
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i=0; i < 100; i++ ){
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			expOut[i] = 1.0 / (1.0 + FastMath.exp(-x));
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softplus", z).derivative());
		
		for( int i=0; i<100; i++ ){
			double relError = Math.abs(expOut[i]-zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
			assertTrue(relError < REL_ERROR_TOLERANCE);
		}
	}
	
	@Test
	public void testTanhDerivative(){
		assertTrue(Nd4j.getOpFactory().createTransform("tanh", Nd4j.ones(1)).derivative() instanceof TanhDerivative);
		
		//Derivative of sigmoid: ds(x)/dx = s(x)*(1-s(x))
		//s(x) = 1 / (exp(-x) + 1)
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i = 0; i < 100; i++) {
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			double tanh = FastMath.tanh(x);
			expOut[i] = 1.0 - tanh * tanh;
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", z).derivative());
		
		for( int i = 0; i < 100; i++ ){
			double relError = Math.abs(expOut[i]-zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
			assertTrue(relError < REL_ERROR_TOLERANCE);
		}
	}
	
	@Test
	public void testLeakyReLUDerivative(){
		assertTrue(Nd4j.getOpFactory().createTransform("leakyrelu", Nd4j.ones(1)).derivative() instanceof LeakyReLUDerivative);
		
		//Derivative: 0.01 if x<0, 1 otherwise
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i = 0; i < 100; i++) {
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			expOut[i] = (x >= 0 ? 1 : 0.01);
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("leakyrelu", z).derivative());
		
		for( int i = 0; i < 100; i++ ){
			double relError = Math.abs(expOut[i]-zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
			assertTrue(relError < REL_ERROR_TOLERANCE);
		}
	}
	
	@Test
	public void testSoftSignDerivative(){
		assertTrue( Nd4j.getOpFactory().createTransform("softsign", Nd4j.ones(1)).derivative() instanceof SoftSignDerivative );
		
		//Derivative: 1 / (1+abs(x))^2
		INDArray z = Nd4j.zeros(100);
		double[] expOut = new double[100]; 
		for( int i = 0; i < 100; i++) {
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			double temp = 1 + Math.abs(x);
			expOut[i] = 1.0 / (temp*temp);
		}
		
		INDArray zPrime = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softsign", z).derivative());
		
		for( int i = 0; i < 100; i++ ){
			double relError = Math.abs(expOut[i] - zPrime.getDouble(i)) / (Math.abs(expOut[i]) + Math.abs(zPrime.getDouble(i)));
			assertTrue(relError < REL_ERROR_TOLERANCE);
		}
	}

	@Test
	public void testELUDerivative(){
		assertTrue( Nd4j.getOpFactory().createTransform("elu",Nd4j.ones(1)).derivative() instanceof ELUDerivative);

		//f(x) = x if x>=0
		//f(x) = 1.0*(exp(x)-1)
		INDArray z = Nd4j.zeros(100);
		double[] out = new double[100];
		double[] outDeriv = new double[100];
		for( int i = 0; i < 100; i++) {
			double x = 0.1 * (i - 50);
			z.putScalar(i, x);
			if(x>=0){
				out[i] = x;
				outDeriv[i] = 1.0;
			} else {
				out[i] = FastMath.exp(x)-1.0;
				outDeriv[i] = FastMath.exp(x);
			}
		}

		INDArray act = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("elu", z.dup()));
		INDArray actDeriv = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("elu", z.dup()).derivative());

		System.out.println(act);

		for( int i = 0; i < 100; i++ ){
			double relError1 = Math.abs(out[i] - act.getDouble(i)) / (Math.abs(out[i]) + Math.abs(act.getDouble(i)));
			if(out[i] == 0.0 && act.getDouble(i) == 0.0) relError1 = 0.0;
			double relError2 = Math.abs(outDeriv[i] - actDeriv.getDouble(i)) / (Math.abs(outDeriv[i]) + Math.abs(actDeriv.getDouble(i)));
			if(outDeriv[i] == 0.0 && actDeriv.getDouble(i) == 0.0) relError2 = 0.0;
			assertTrue(relError1 < REL_ERROR_TOLERANCE);
			assertTrue(relError2 < REL_ERROR_TOLERANCE);
		}
	}

    @Override
    public char ordering() {
        return 'f';
    }
}
