package nilgiri.processing;


import processing.core.PApplet;

public class MouseGestureTranslationMode extends MouseGestureMode{
	private PApplet m_applet;
	private ViewConfig2D m_vc; 
		
	public MouseGestureTranslationMode(PApplet i_applet, ViewConfig2D i_vc){
		super();
		m_applet = i_applet;
		m_vc = i_vc;
	}

	public void mouseDragged(){
		float[] translation = new float[2];
		m_vc.getTranslation(translation);
		float scale = m_vc.getScale();
		translation[0] += (m_applet.mouseX - m_applet.pmouseX)/scale;
		translation[1] += (m_applet.mouseY - m_applet.pmouseY)/scale;
		m_vc.setTranslation(translation);
	}

	public String toString(){
		return "Mode [Translation]";
	}
}
