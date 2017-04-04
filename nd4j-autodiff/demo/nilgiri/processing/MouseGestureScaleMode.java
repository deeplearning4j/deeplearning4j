package nilgiri.processing;


import processing.core.PApplet;

public class MouseGestureScaleMode extends MouseGestureMode{
	private PApplet m_applet;
	private ViewConfig2D m_vc; 
		
	public MouseGestureScaleMode(PApplet i_applet, ViewConfig2D i_vc){
		super();
		m_applet = i_applet;
		m_vc = i_vc;
	}

	public void mouseDragged(){
		float dscale = 0.1f*(m_applet.mouseY - m_applet.pmouseY);
		m_vc.setScale(m_vc.getScale()*(1.0f + dscale));
	}

	public String toString(){
		return "Mode [Zoom]";
	}
}
