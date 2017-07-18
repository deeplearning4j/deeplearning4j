# [JAutoDiff](http://uniker9.github.com/JAutoDiff/) : An Automatic Differentiation Library (Pure Java)

[Automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation/) 
is a technique to compute the derivatives of functions algebraically.  
There are many implementation in C/C++ (ex. 
[FADBAD++](http://www.fadbad.com/fadbad.html), 
[ADOL-C](http://projects.coin-or.org/ADOL-C), etc.), while are few in Java.

*JAutoDiff* is an automatic differentiation library coded in 100%  pure Java.  
This library provides a framework to compute derivatives of functions 
on arbitrary types of [field](http://en.wikipedia.org/wiki/Field_\(mathematics\)) 
using [generics](http://en.wikipedia.org/wiki/Generics_in_Java).

This software is released under the MIT License, see LICENSE.txt.	

# Sample : [ADTest.java](https://github.com/uniker9/JAutoDiff/blob/master/JAutoDiff/test/nilgiri/math/autodiff/ADTest.java)

	package nilgiri.math.autodiff;
	
	import org.junit.Test;
	
	
	import junit.framework.Assert;
	import nilgiri.math.DoubleReal;
	import nilgiri.math.DoubleRealFactory;
	
	public class ADTest {
		private final DoubleRealFactory RNFactory = DoubleRealFactory.instance();
		private final DifferentialRealFunctionFactory<DoubleReal> DFFactory = new DifferentialRealFunctionFactory<DoubleReal>(RNFactory);
	
		private void test(double i_expected, DifferentialFunction<DoubleReal> i_f){
			String func_str = i_f.toString();
			double func_value = i_f.getValue().doubleValue();
			
			System.out.println(func_str +" = "+ func_value +" is expected as "+ i_expected);
			Assert.assertEquals(func_str, i_expected, func_value);
		}
	
		@Test
		public void testFunc() {
			double vx = 3.0;
			double vy = 5.0;
			double vq = 8.0;
	
			Variable<DoubleReal> x = DFFactory.var("x", new DoubleReal(vx));
			Variable<DoubleReal> y = DFFactory.var("y", new DoubleReal(vy));
			Constant<DoubleReal> q = DFFactory.val(new DoubleReal(vq));
	
			//================================================================================
			//Construct functions
			//================================================================================
			//h = q*x*( cos(x*y) + y )
			DifferentialFunction<DoubleReal> h = q.mul(x).mul( DFFactory.cos( x.mul(y) ).plus(y) );
			
			//ph/px = q*( cos(x*y) + y ) + q*x*( -sin(x*y)*y ) 
			DifferentialFunction<DoubleReal> dhpx = h.diff(x);
	
			//ph/py = q*x*( -sin(x*y)*x + 1.0) 
			DifferentialFunction<DoubleReal> dhpy = h.diff(y);
			
			//p2h/px2 = q*( -sin(x*y)*y + y ) + q*( -sin(x*y)*y ) + q*x*( -cos(x*y)*y*y ) 
			DifferentialFunction<DoubleReal> d2hpxpx = dhpx.diff(x);
	
			//p2h/pypx = q*( -sin(x*y)*x + 1.0 ) + q*x*( -sin(x*y) - cos(x*y)*y*y ) 
			DifferentialFunction<DoubleReal> d2hpypx = dhpx.diff(y);
	
			//================================================================================
			//Test functions { h, ph/px, ph/py, p2h/px2, p2h/pypx }.
			//================================================================================
			test(vq*vx*( Math.cos(vx*vy) + vy ), h);
			test(vq*( Math.cos(vx*vy) + vy ) + vq*vx*(-Math.sin(vx*vy)*vy ), dhpx);
			test(vq*vx*( -Math.sin(vx*vy)*vx + 1.0 ), dhpy);
			test(vq*( -Math.sin(vx*vy)*vy ) + vq*( -Math.sin(vx*vy)*vy ) + vq*vx*(-Math.cos(vx*vy)*vy*vy), d2hpxpx);
			test(vq*( -Math.sin(vx*vy)*vx + 1.0 ) + vq*vx*( -Math.sin(vx*vy) - Math.cos(vx*vy)*vx*vy ), d2hpypx);
	
			//================================================================================
			//Change values of the variables.
			//================================================================================
			vx = 4.0;
			vy = 7.0;
			x.set(new DoubleReal(vx));
			y.set(new DoubleReal(vy));
	
			//================================================================================
			//Re-Test functions { h, ph/px, ph/py, p2h/px2, p2h/pypx }.
			//================================================================================
			//No reconstruction of the functions is necessary 
			//to get values of the functions for new values of variables.
			test(vq*vx*( Math.cos(vx*vy) + vy ), h);
			test(vq*( Math.cos(vx*vy) + vy ) + vq*vx*(-Math.sin(vx*vy)*vy ), dhpx);
			test(vq*vx*( -Math.sin(vx*vy)*vx + 1.0 ), dhpy);
			test(vq*( -Math.sin(vx*vy)*vy ) + vq*( -Math.sin(vx*vy)*vy ) + vq*vx*(-Math.cos(vx*vy)*vy*vy), d2hpxpx);
			test(vq*( -Math.sin(vx*vy)*vx + 1.0 ) + vq*vx*( -Math.sin(vx*vy) - Math.cos(vx*vy)*vx*vy ), d2hpypx);
	
		}
	}
	

## Result

	((8.0*x)*(cos((x*y))+y)) = 101.7674900913883 is expected as 101.7674900913883
	((8.0*(cos((x*y))+y))+((8.0*x)*-(sin((x*y))*y))) = -44.112044121724594 is expected as -44.112044121724594
	((8.0*x)*(-(sin((x*y))*x)+1.0)) = -22.82072449131241 is expected as -22.82072449131241
	((8.0*-(sin((x*y))*y))+((8.0*-(sin((x*y))*y))+((8.0*x)*-((cos((x*y))*y)*y)))) = 403.78972050272347 is expected as 403.78972050272347
	((8.0*(-(sin((x*y))*x)+1.0))+((8.0*x)*-(((cos((x*y))*x)*y)+sin((x*y))))) = 250.27383230163406 is expected as 250.27383230163406
	((8.0*x)*(cos((x*y))+y)) = 193.19661227796587 is expected as 193.19661227796587
	((8.0*(cos((x*y))+y))+((8.0*x)*-(sin((x*y))*y))) = -12.383743511471195 is expected as -12.383743511471195
	((8.0*x)*(-(sin((x*y))*x)+1.0)) = -2.6759409034072377 is expected as -2.6759409034072377
	((8.0*-(sin((x*y))*y))+((8.0*-(sin((x*y))*y))+((8.0*x)*-((cos((x*y))*y)*y)))) = 1479.024550089191 is expected as 1479.024550089191
	((8.0*(-(sin((x*y))*x)+1.0))+((8.0*x)*-(((cos((x*y))*x)*y)+sin((x*y))))) = 853.1568857652521 is expected as 853.1568857652521

