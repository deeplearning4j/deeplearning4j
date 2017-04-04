package nilgiri.physics;

import nilgiri.math.AbstractRealNumberFactory;
import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.Constant;
import nilgiri.math.autodiff.DifferentialFunction;
import nilgiri.math.autodiff.DifferentialRealFunctionFactory;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.autodiff.Variable;

public class OrbitFactory<X extends RealNumber<X>> {
	
	
	private AbstractRealNumberFactory<X> m_VF; 	
	private	DifferentialRealFunctionFactory<X> m_FF;
	
	public OrbitFactory(AbstractRealNumberFactory<X> i_VF, 
			DifferentialRealFunctionFactory<X> i_FF){
		m_VF = i_VF;
		m_FF = i_FF;
	}
	
	public DifferentialVectorFunction<X> createCircularOrbit(
			DifferentialVectorFunction<X> i_origin,
			Variable<X> i_t,
			X i_a,
			X i_omega,
			X i_t0,
			X i_theta0){
	
		Constant<X> a = m_FF.val(i_a); 
		DifferentialFunction<X> theta = m_FF.val(i_theta0).plus(m_FF.val(i_omega).mul(i_t.minus(m_FF.val(i_t0))));
		
		return m_FF.function(
				i_origin.get(0).plus(a.mul(m_FF.cos(theta))),
				i_origin.get(1).plus(a.mul(m_FF.sin(theta))));
	};
		
}
