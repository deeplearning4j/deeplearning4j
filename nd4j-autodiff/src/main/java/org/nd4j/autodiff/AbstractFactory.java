package org.nd4j.autodiff;


import org.nd4j.autodiff.graph.graph.Graph;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;

public interface AbstractFactory<X extends Field<X>>
        extends AbstractIdentityFactory<X> {


 Graph<NDArrayInformation,OpState> graph();

   /* Add scalar ops here */

 X add(X i_x,Number value);
 X sub(X i_x,Number value);
 X mul(X i_x,Number value);
 X div(X i_x,Number value);

 X broadcast(X i_x,int...shape);

 X repeat(X i_x,int axis);

 X tile(X i_x,int...repeat);

 X sum(X i_x,int...dimensions);
 X prod(X i_x,int...dimensions);
 X mean(X i_x,int...dimensions);
 X std(X i_x,int...dimensions);
 X variance(X i_x,int...dimensions);
 X max(X i_x,int...dimensions);
 X min(X i_x,int...dimensions);
 X norm1(X i_x,int...dimensions);
 X norm2(X i_x,int...dimensions);
 X normmax(X i_x,int...dimensions);


 X transpose(X i_x);

 X reshape(X i_x, int[] shape);


 X valueArrayOf(X i_x,int[] shape);

 X val(double i_v);

 X abs(X i_x);

 X min(X i_x, X i_y);
 X max(X i_x, X i_y);

 X cos(X i_x);
 X acos(X i_x);
 X cosh(X i_x);
 X acosh(X i_x);

 X sin(X i_x);
 X asin(X i_x);
 X sinh(X i_x);
 X asinh(X i_x);

 X tan(X i_x);
 X atan(X i_x);
 X atan2(X i_x, X i_y);
 X tanh(X i_x);
 X atanh(X i_x);

 X exp(X i_x);
 X log(X i_x);

 X log10(X i_x);

 X flat(X i_x);
 X mc(X i_x, X i_y);
 X rand(X i_x);
 X random(X i_x);
 X gauss(X i_x);

 X sgn(X i_x);

 X ifx(X i_x, X i_y, X i_z);

 X buf(X i_x);
 X inv(X i_x);
 X u(X i_x);
 X uramp(X i_x);

 X pow(X i_x, X i_y);
 X pwr(X i_x, X i_y);
 X pwrs(X i_x, X i_y);
 X sqrt(X i_x);
 X square(X i_x);
 X hypot(X i_x, X i_y);

 X floor(X value);
 X ceil(X value);
 X round(X value);

 X relu(X value);

 X leakyRelu(X value,double alpha);

 /**
  * Leaky relu with an alpha of
  * 0.01
  * @param value the value to transform
  * @return
  */
 X leakyRelu(X value);

 X leakyReluDerivative(X value,double alpha);

 /**
  * Leaky relu with an alpha of
  * 0.01
  * @param value the value to transform
  * @return
  */
 X leakyReluDerivative(X value);


 X hardTanh(X value);

 X hardTanhDerivative(X value);

 X sigmoid(X value);

 X sigmoidDerivative(X value);


 X softmax(X value);

 X elu(X value);

 X eluDerivative(X value);

 X step(X value);

 X sign(X value);

 X softsign(X value);

 X softsignDeriviative(X value);

 X softplus(X value);

}
