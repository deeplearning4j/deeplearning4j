package nilgiri.physics;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialFunction;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.autodiff.Variable;
import nilgiri.math.autodiff.VariableVector;

public class AccelerateMover<R extends RealNumber<R>> extends AbstractMover<R>{ 
	protected DifferentialFunction<R> m_radius;
	protected VariableVector<R> m_position;
	protected VariableVector<R> m_velocity;
	protected DifferentialVectorFunction<R> m_accel;
	
	public AccelerateMover(
			Variable<R> i_t,
			DifferentialFunction<R> i_radius,
			VariableVector<R> i_position,
			VariableVector<R> i_velocity,
			DifferentialVectorFunction<R> i_accel){
		super(i_t);
		m_radius = i_radius;
		m_position = i_position;
		m_velocity = i_velocity;
		m_accel = i_accel;
	}

	@Override
	public DifferentialFunction<R> radius() {
		return m_radius;
	}
	
	@Override
	public VariableVector<R> position() {
		return m_position;
	}
	
	@Override
	public VariableVector<R> velocity() {
		return m_velocity;
	}
	
	@Override
	public DifferentialVectorFunction<R> acceleration() {
		return m_accel;
	}
}
