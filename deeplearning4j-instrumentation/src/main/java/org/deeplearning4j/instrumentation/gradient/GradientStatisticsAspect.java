package org.deeplearning4j.instrumentation.gradient;

import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Aspect;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Aspect
public class GradientStatisticsAspect {

	private static Logger log = LoggerFactory.getLogger(GradientStatisticsAspect.class);


	
	@AfterReturning(pointcut= "call(org.deeplearning4j.nn.gradient.NeuralNetworkGradient *.*(..))",returning="gradient")
	public void intercept(Object gradient) throws Throwable {
		System.out.println("Got a gradient");
	}



}
