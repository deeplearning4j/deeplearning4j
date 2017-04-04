package nilgiri.planarSolarSystem;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import nilgiri.math.AbstractRealNumberFactory;
import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.Constant;
import nilgiri.math.autodiff.DifferentialRealFunctionFactory;
import nilgiri.math.autodiff.Variable;
import nilgiri.physics.AbstractMover;
import nilgiri.physics.AnalyticalMover;

public class MoverSimulator<X extends RealNumber<X>> {
	
	
	private final AbstractRealNumberFactory<X> VF;
	private final DifferentialRealFunctionFactory<X> FF;

	
	private Variable<X> m_t;
	private Constant<X> m_dt;
	
	private ArrayList<AnalyticalMover<X>> m_analiticalMoverList;
	private HashMap<AbstractMover<X>, String> m_moverNameMap; 
	
	public MoverSimulator(
			AbstractRealNumberFactory<X> i_VF,
			DifferentialRealFunctionFactory<X> i_FF
			){
		VF = i_VF;
		FF = i_FF;

		
		m_t = FF.var("t", VF.zero());
		m_dt = FF.val(VF.zero());
		
		m_analiticalMoverList = new ArrayList<AnalyticalMover<X>>(10);
//		m_accelerateMoverList = new ArrayList<AccelerateMover<X>>(10);
		
		m_moverNameMap = new HashMap<AbstractMover<X>, String>(4);
	}
	
	public void addAnalyticalMover(String i_name, AnalyticalMover i_mover){
		m_moverNameMap.put(i_mover, i_name);
		m_analiticalMoverList.add(i_mover);
	}
	
	
	public void setT(X i_t){
		m_t.set(i_t);
	}
	public void setDT(X i_dt){
		m_dt = FF.val(i_dt);
	}
	
	
	public void nextStep(){
		Constant<X> dt = getDT();
		
		
		m_t.set(getT().getValue().plus(dt.getValue()));

		
		/*
		Iterator<AccelerateMover<X>> itrA =  getMoverAIterator();
		while(itrA.hasNext()){
			AccelerateMover<X> mover = itrA.next();
			
			mover.velocity().assign(mover.velocity().plus(mover.accel().mul(dt)));
			mover.position().assign(mover.position().plus(mover.velocity().mul(dt)));

		}
		*/
	}
	
	
	public Variable<X> getT(){
		return m_t;
	}
	public Constant<X> getDT(){
		return m_dt;
	}
	public String getNameOf(AbstractMover<X> i_mover){
		return m_moverNameMap.get(i_mover);
	}
	

	public Iterator<AnalyticalMover<X>> getAnalitycalMoverIterator(){
		return m_analiticalMoverList.iterator();
	}
//	public Iterator<AccelerateMover<X>> getAccelerateMoverIterator(){
//		return m_accelerateMoverList.iterator();
//	}

	
	public Iterator<AbstractMover<X>> getMoverIterator(){
		return m_moverNameMap.keySet().iterator();		
	}
			
	
	
	
	
}
