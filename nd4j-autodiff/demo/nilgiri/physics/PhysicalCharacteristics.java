package nilgiri.physics;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialFunction;

public class PhysicalCharacteristics <X extends RealNumber<X>> 
{
	private DifferentialFunction<X> m_radius;
	private DifferentialFunction<X> m_mass;
	
	
	public PhysicalCharacteristics(DifferentialFunction<X> i_radius,
			DifferentialFunction<X> i_mass){
		m_radius = i_radius;
		m_mass = i_mass;
	}
	
	public DifferentialFunction<X> mass(){
		return m_mass;
	}
	public DifferentialFunction<X> radius(){
		return m_radius;
	}
}
