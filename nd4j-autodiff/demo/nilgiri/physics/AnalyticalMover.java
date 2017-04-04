package nilgiri.physics;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialFunction;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.autodiff.Variable;


public class AnalyticalMover<R extends RealNumber<R>> extends AbstractMover<R>{

	protected DifferentialFunction<R> m_radius;
	protected DifferentialVectorFunction<R> m_position;
	protected DifferentialVectorFunction<R> m_velocity;
	protected DifferentialVectorFunction<R> m_accel;
	
	public AnalyticalMover(
			Variable<R> i_t,
			DifferentialFunction<R> i_radius,
			DifferentialVectorFunction<R> i_pos){
		super(i_t);
		m_radius = i_radius;
		m_position = i_pos;
		m_velocity = m_position.diff(m_t);
		m_accel = m_velocity.diff(m_t);
	}

	@Override
	public DifferentialFunction<R> radius() {
		return m_radius;
	}

	@Override
	public DifferentialVectorFunction<R> position() {
		return m_position;
	}

	@Override
	public DifferentialVectorFunction<R> velocity() {
		return m_velocity;
	}

	@Override
	public DifferentialVectorFunction<R> acceleration() {
		return m_accel;
	}
	
}
