package nilgiri.physics;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialFunction;

public class OrbitalElements <X extends RealNumber<X>> {

	private DifferentialFunction<X> m_epoch;
	private DifferentialFunction<X> m_e;
	private DifferentialFunction<X> m_a;
	private DifferentialFunction<X> m_inclination;
	private DifferentialFunction<X> m_node;
	private DifferentialFunction<X> m_perigee;
	private DifferentialFunction<X> m_manomaly;
	
	public OrbitalElements(
			DifferentialFunction<X> i_epoch, //epoch
			DifferentialFunction<X> i_a, //semi-major axis 
			DifferentialFunction<X> i_e, //eccentricity
			DifferentialFunction<X> i_inclination, // inclination. 0 for 2D
			DifferentialFunction<X> i_node, // right ascension of the ascending node.  0 for 2D
			DifferentialFunction<X> i_perigee, //argument of perigee
			DifferentialFunction<X> i_manomaly //mean anomaly
			){
		m_epoch = i_epoch;
		m_a = i_a;
		m_e = i_e;
		m_inclination = i_inclination;
		m_node = i_node;
		m_perigee = i_perigee;
		m_manomaly = i_manomaly;
	}
	
	public DifferentialFunction<X> epoch(){
		return m_epoch;
	}  
	public DifferentialFunction<X> a(){
		return m_a;
	}  
	public DifferentialFunction<X> e(){
		return m_e;
	}  
	public DifferentialFunction<X> inclination(){
		return m_inclination;
	}  
	public DifferentialFunction<X> node(){
		return m_node;
	}  
	public DifferentialFunction<X> perigee(){
		return m_perigee;
	}  
	public DifferentialFunction<X> meanAnomaly(){
		return m_manomaly;
	}  

	
}
