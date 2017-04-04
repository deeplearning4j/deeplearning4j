package nilgiri.planarSolarSystem;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.physics.AbstractMover;

public class StateGetter<R extends RealNumber<R>> {
	private StateType m_stateType; 
	private InternalGetter<R> m_getter;
	


	public StateGetter(StateType i_type){
		setStateType(i_type);
	}
	
	public StateType getStateType(){
		return m_stateType;
	}
	public void setStateType(StateType i_stateType){
		switch(i_stateType){
		case POSITION:
			m_getter = new PositionGetter<R>();
			break;
		case VELOCITY:
			m_getter = new VelocityGetter<R>();
			break;
		case ACCELERATION:
			m_getter = new AccelerationGetter<R>();
			break;
		default:
			i_stateType = StateType.POSITION;
			m_getter = new PositionGetter<R>();
			break;
		}
		m_stateType = i_stateType;
	}
	
	public DifferentialVectorFunction<R> getState(AbstractMover<R> i_mover){
		return m_getter.getState(i_mover);
	}
	
	
	private abstract class InternalGetter<R extends RealNumber<R>>{
		abstract public DifferentialVectorFunction<R> getState(AbstractMover<R> i_mover);
	}

	private class PositionGetter<R extends RealNumber<R>> extends InternalGetter<R>{
		public DifferentialVectorFunction<R> getState(AbstractMover<R> i_mover){
			return i_mover.position();
		}
	}
	private class VelocityGetter<R extends RealNumber<R>> extends InternalGetter<R>{
		public DifferentialVectorFunction<R> getState(AbstractMover<R> i_mover){
			return i_mover.velocity();
		}
	}
	private class AccelerationGetter<R extends RealNumber<R>> extends InternalGetter<R>{
		public DifferentialVectorFunction<R> getState(AbstractMover<R> i_mover){
			return i_mover.acceleration();
		}
	}


}
