package org.deeplearning4j.transformation;

public class MatrixTransformations {

	public static MatrixTransform multiplyScalar(double num) {
		return new MultiplyScalar(num);
	}
	
	public static MatrixTransform addScalar(double num) {
		return new AddScalar(num);
	}
	
	public static MatrixTransform divideScalar(double num) {
		return new DivideScalar(num);
	}
	
	
	public static MatrixTransform sqrt() {
		return new SqrtScalar();
	}
	
	public static MatrixTransform exp() {
		return new ExpTransform();
	}
	
	public static MatrixTransform log() {
		return new LogTransform();
	}
	
	
	public static MatrixTransform powScalar(double num) {
		return new PowScale(num);
	}

}
