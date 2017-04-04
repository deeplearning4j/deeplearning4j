package nilgiri.physics;

import java.util.ArrayList;

import java.util.Iterator;
import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialFunction;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.autodiff.Variable;

abstract public class AbstractMover<R extends RealNumber<R> >{

	protected Variable<R> m_t;
	
	AbstractMover(Variable<R> i_t){
		m_t = i_t;
	}

	public Variable<R> t(){
		return m_t;
	}
	
	abstract public DifferentialFunction<R> radius();
	abstract public DifferentialVectorFunction<R> position();
	abstract public DifferentialVectorFunction<R> velocity();
	abstract public DifferentialVectorFunction<R> acceleration();
	

//
	
}
