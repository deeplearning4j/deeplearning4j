package nilgiri.processing;
import java.awt.BorderLayout;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

import processing.core.PApplet;


public class PAppletFrame extends JFrame{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private PApplet m_applet;
	
	public PAppletFrame(PApplet i_applet){
		super();
		m_applet = i_applet;
		
		addWindowListener(new WindowAdapter(){
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});

		addComponentListener(new ComponentAdapter(){
			public void componentShown(ComponentEvent e) {
				if(m_applet!=null){
					m_applet.redraw();
				}
			}
			public void componentResized(ComponentEvent e) {
				if(m_applet!=null){
					if(m_applet.g.is3D()){
						m_applet.size(m_applet.getWidth(), m_applet.getHeight(), PApplet.P3D); //3D
					}else{
						m_applet.size(m_applet.getWidth(), m_applet.getHeight()); //2D
					}
					m_applet.redraw();
				}
			}
		});
		setLayout(new BorderLayout());
		add(m_applet,BorderLayout.CENTER);
	}
}
