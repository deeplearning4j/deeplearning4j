package nilgiri.planarSolarSystem;

import java.awt.Color;

import nilgiri.math.RealNumber;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.ia.DoubleRealInterval;
import nilgiri.processing.AbstractViewerPApplet;
import processing.core.PApplet;

public class PSSMoverDrawer implements Drawer<PSSMover<DoubleRealInterval>> {
	
	private AbstractViewerPApplet m_applet;
	private StateGetter<DoubleRealInterval> m_getter;
	
	
	
	public PSSMoverDrawer(AbstractViewerPApplet i_applet, StateGetter<DoubleRealInterval> i_getter){
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
		
		
		m_applet.fill(i_mover.color().getRGB());
		m_applet.ellipseMode(PApplet.RADIUS);
		m_applet.ellipse(0.0f, 0.0f, i_mover.rx(), i_mover.ry());
		
		
		
		float w = (float)(scale*x.width());
		float h = (float)(scale*y.width());
		m_applet.noFill();
		m_applet.rect(-0.5f*w, -0.5f*h, w, h);
		m_applet.popMatrix();
		
	}

}
