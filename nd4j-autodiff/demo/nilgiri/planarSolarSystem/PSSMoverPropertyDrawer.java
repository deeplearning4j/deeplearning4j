package nilgiri.planarSolarSystem;

import java.awt.Color;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.ia.DoubleRealInterval;
import nilgiri.processing.AbstractViewerPApplet;
import processing.core.PApplet;

public class PSSMoverPropertyDrawer implements Drawer<PSSMover<DoubleRealInterval>> {
	
	private AbstractViewerPApplet m_applet;
	private StateGetter<DoubleRealInterval> m_getter;
	
	
	
	public PSSMoverPropertyDrawer(AbstractViewerPApplet i_applet, StateGetter<DoubleRealInterval> i_getter){
		m_applet = i_applet;
		m_getter = i_getter;
		
	}
	
	
	
	public StateGetter<DoubleRealInterval> stateGetter(){
		return m_getter;
	}
	
	
	public void draw(PSSMover<DoubleRealInterval> i_mover) {
		DifferentialVectorFunction<DoubleRealInterval> state = m_getter.getState(i_mover.mover());
		DoubleRealInterval x = state.get(0).getValue();
		DoubleRealInterval y = state.get(1).getValue();
		
		
		m_applet.pushMatrix();
		float scale = m_applet.viewConfig().getScale();
		m_applet.translate((float)(scale*x.center()), (float)(scale*-y.center()));
		m_applet.fill(Color.WHITE.getRGB());
		m_applet.text(i_mover.name(), 0, 0);
//		m_applet.text(i_mover.name() + String.format("([%6.0f +- %6.5f], [%6.0f +- %6.5f])", x.center(), 0.5*x.width(), y.center(), 0.5*y.width()), 0, 0);
		
		
		m_applet.popMatrix();
		
	}

}
