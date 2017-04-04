package nilgiri.planarSolarSystem;

import java.awt.Color;
import nilgiri.math.RealNumber;
import nilgiri.physics.AbstractMover;


public class PSSMover<R extends RealNumber<R>> implements Drawable{

	
	private String m_name;
	private AbstractMover<R> m_mover;
	private Color m_color;
	private float m_rx;
	private float m_ry;

	
	public PSSMover(String i_name,  AbstractMover<R> i_mover, Color i_color, float i_r){
		this(i_name, i_mover, i_color, i_r, i_r);
	}
	public PSSMover(String i_name,  AbstractMover<R> i_mover, Color i_color, float i_rx, float i_ry){
		m_name = i_name;
		m_mover = i_mover;
		
		m_color = i_color;
		m_rx = i_rx;
		m_ry = i_ry;
	}
	
	
	public String name(){
		return m_name;
	}
	public AbstractMover<R> mover(){
		return m_mover;
	}
	public Color color(){
		return m_color;
	}
	public float rx(){
		return m_rx;
	}
	public float ry(){
		return m_ry;
	}
	
}
