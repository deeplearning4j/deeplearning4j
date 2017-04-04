package nilgiri.planarSolarSystem;



import java.awt.Color;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.Iterator;
import javax.swing.JFrame;

import nilgiri.math.autodiff.DifferentialRealFunctionFactory;
import nilgiri.math.autodiff.DifferentialVectorFunction;
import nilgiri.math.autodiff.Variable;
import nilgiri.math.ia.DoubleRealInterval;
import nilgiri.math.ia.DoubleRealIntervalFactory;
import nilgiri.physics.AnalyticalMover;
import nilgiri.physics.OrbitFactory;
import nilgiri.processing.AbstractViewerPApplet;
import nilgiri.processing.PAppletFrame;
import nilgiri.processing.ViewConfig2D;

import processing.core.PApplet;

public class PlanarSolarSystemApplet extends AbstractViewerPApplet{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private MoverSimulator<DoubleRealInterval> m_simulator; // must be initialized in setup()
	private int m_steps = 0;
	

	private StateType m_sstype = StateType.NONE;
	private String m_sstypeName = "";
	

	
	private StateGetter<DoubleRealInterval> m_stateGetter;
	protected ArrayList<Drawer<PSSMover<DoubleRealInterval>>> m_drawerList = new ArrayList<Drawer<PSSMover<DoubleRealInterval>>>() ;
	protected ArrayList<PSSMover<DoubleRealInterval>> m_moverList = new ArrayList<PSSMover<DoubleRealInterval>>();
	
	
	private static final double AU = 149597870000.0; // Astronomical Unit : the mean Earth-Sun distance.

	
	
	public StateType getStateSpaceType(){
		return m_sstype;
	}
	public void setStateSpaceType(StateType i_s){
		ViewConfig2D vc = viewConfig();
		switch(i_s){
		case POSITION:
			m_sstype = StateType.POSITION;
			m_stateGetter.setStateType(StateType.POSITION);
			m_sstypeName = "State [Position]";
			vc.setScale((float)(0.3f*this.getWidth()/AU));
			vc.setTranslation(0.0f, 0.0f);
			break;
		case VELOCITY:
			m_sstype = StateType.VELOCITY;
			m_stateGetter.setStateType(StateType.VELOCITY);
			m_sstypeName = "State [Velocity]";
			vc.setScale(1e-3f);
			vc.setTranslation(0.0f, 0.0f);
			break;
		case ACCELERATION:
			m_sstype = StateType.ACCELERATION;
			m_stateGetter.setStateType(StateType.ACCELERATION);
			m_sstypeName = "State [Acceleration]";
			vc.setScale(1e+3f);
			vc.setTranslation(0.0f, 0.0f);
			break;
		default:
			break;
		}
	}

	
	
	public void setup(){
		
		int windowWidth = 600;
		int windowHeight = 600;

		size(windowWidth, windowHeight);
		textFont(createFont("Lucida Console", 12));
				
		Color bgcolor = new Color(0, 0, 0);
		this.setBackground(bgcolor);
		background(bgcolor.getRGB());
		
		stroke(255);
		strokeWeight(0.1f);
		
		//----------------------------------------
		// Setup Simulation
		//----------------------------------------
		DoubleRealIntervalFactory VF = DoubleRealIntervalFactory.instance();
		DifferentialRealFunctionFactory<DoubleRealInterval> FF = new DifferentialRealFunctionFactory<DoubleRealInterval>(VF);
		m_simulator = new MoverSimulator<DoubleRealInterval>(VF, FF);
		m_simulator.setT(VF.valWithTolerance(0.0, 50000.0));
		m_simulator.setDT(VF.val(10000.0));
		
		
		Variable<DoubleRealInterval> t = m_simulator.getT();
		
		OrbitFactory<DoubleRealInterval> MF = new OrbitFactory<DoubleRealInterval>(VF, FF);
		
		final int DIM = 2;
		
		m_stateGetter = new StateGetter<DoubleRealInterval>(StateType.POSITION);
		setStateSpaceType(StateType.POSITION);
		m_drawerList.add(new PSSMoverDrawer(this, m_stateGetter));
		m_drawerList.add(new PSSMoverPropertyDrawer(this, m_stateGetter));
		
		//----------------------------------------
		// Create Sun
		//----------------------------------------
		DifferentialVectorFunction<DoubleRealInterval> sun_orbit = MF.createCircularOrbit(	
				FF.zero(DIM),
				t,
				VF.zero(), //a
				VF.zero(), //omega
				VF.zero(), //t0
				VF.zero() //theta(t0)
				);
		AnalyticalMover<DoubleRealInterval> sun  = new AnalyticalMover<DoubleRealInterval>(
				t, FF.val(VF.val(1392000000.0)), sun_orbit);
		PSSMover<DoubleRealInterval> pss_sun = new PSSMover<DoubleRealInterval>("Sun", sun, new Color(255, 69, 0), 20.0f);
		m_moverList.add(pss_sun);
		m_simulator.addAnalyticalMover("Sun", sun);
		
		
		//----------------------------------------
		// Create Earth
		//----------------------------------------
		DifferentialVectorFunction<DoubleRealInterval> earth_orbit
		= MF.createCircularOrbit(	
				sun.position(),
				t,
				VF.val(AU), //Semi-major axis [m]
				VF.val(2.0*Math.PI/(365.0*24.0*3600.0)), //  2-Pi / (Orbital Period) [rad/s]
				VF.zero(), //t0
				VF.zero() //theta(t0)
				);
		AnalyticalMover<DoubleRealInterval> earth  = new AnalyticalMover<DoubleRealInterval>(
				t, FF.val(VF.val(6356752.0)), earth_orbit);
		PSSMover<DoubleRealInterval> pss_earth = new PSSMover<DoubleRealInterval>("Earth", earth, new Color(0, 0, 205), 5.0f);
		m_moverList.add(pss_earth);
		m_simulator.addAnalyticalMover("Earth", earth);
		
		//----------------------------------------
		// Create Moon
		//----------------------------------------
		DifferentialVectorFunction<DoubleRealInterval> moon_orbit
		= MF.createCircularOrbit(	
				earth.position(),
				t, 
				VF.val(38440000.0*1000.0), //Semi-major axis [m]
				VF.val(2.0*Math.PI/(30.0*24.0*3600.0)), // 2-Pi / (Orbital Period) [rad/s]
				VF.zero(),
				VF.zero()
				);
		AnalyticalMover<DoubleRealInterval> moon  = new AnalyticalMover<DoubleRealInterval>(
				t, FF.val(VF.val(137150.0)), moon_orbit);
		PSSMover<DoubleRealInterval> pss_moon = new PSSMover<DoubleRealInterval>("Moon", moon, Color.YELLOW, 3.0f);
		m_moverList.add(pss_moon);
		m_simulator.addAnalyticalMover("Moon", moon);
		
		setStepsPerFrame(1);
	}
	
	public void setStepsPerFrame(int i_steps){ //Steps Per Frame
		m_steps = i_steps;
	}
	
	
	
	
	public void draw(){
		background(getBackground().getRGB());
		pushMatrix();
		translate(0.5f*getWidth(), 0.5f*getHeight());
		pushMatrix();
		float[] translation = new float[2];
		ViewConfig2D vc = viewConfig();
		scale(vc.getScale());				
		vc.getTranslation(translation);
		translate(translation[0], translation[1]);
		scale(1.0f/viewConfig().getScale());		
		
	
		
			
		Iterator<PSSMover<DoubleRealInterval> > itr =  m_moverList.iterator();
		while(itr.hasNext()){
			PSSMover<DoubleRealInterval> mover = itr.next();
			Iterator<Drawer<PSSMover<DoubleRealInterval>>> drawer_itr =  m_drawerList.iterator();
			while(drawer_itr.hasNext()){
				Drawer<PSSMover<DoubleRealInterval>> drawer = drawer_itr.next();
				drawer.draw(mover);
			}
		}
		
		popMatrix();
		popMatrix();
		
		// Draw Header Field
		fill(getBackground().getRGB());
		final int header_box_height = 30;
		final int footer_box_height = 30;
		rect(0, 0, getWidth(), header_box_height);
		rect(0, getHeight()-footer_box_height, getWidth(), getHeight());
		fill(255);

		
		final int header_box_padding_left = 20;
		final int header_box_padding_bottom = 10;
		final int header_line = (header_box_height - header_box_padding_bottom);
		text(m_sstypeName, header_box_padding_left, header_line);
		text(this.getMouseGestureMode().toString(), getWidth()-150, header_line);

		DoubleRealInterval t = m_simulator.getT().getValue();
		final double tcenter = t.center();
		final double trange = 0.5*t.width();
		final int footer_box_padding_left = 20;
		final int footer_box_padding_bottom = 10;
		final int footer_line = (getHeight() - footer_box_padding_bottom);
		text("t : "+ String.format("%6.0f +- %6.5f", tcenter, trange), footer_box_padding_left, footer_line);
		
		
		for(int i = 0; i < m_steps; i++){
			m_simulator.nextStep();
			//m_simulator.setT(VF.valWithTolerance(0.0, 50000.0));
			//m_simulator.setDT(VF.val(20000.0));
		}
	}
	
	
	
	
	public void keyPressed(){
		switch(keyCode){
		case KeyEvent.VK_P:
			setStateSpaceType(StateType.POSITION);
			break;
		case KeyEvent.VK_V:
			setStateSpaceType(StateType.VELOCITY);
			break;
		case KeyEvent.VK_A:
			setStateSpaceType(StateType.ACCELERATION);
			break;
		default:
			super.keyPressed();
		}
	}
	
	public static void main(String[] args){

		PApplet applet = new PlanarSolarSystemApplet();
		applet.init();

		

		try{
			JFrame frame = new PAppletFrame(applet);
			frame.pack();
			frame.setLocation(200, 200);
			frame.setVisible(true);
		}catch(Exception ex){
			ex.printStackTrace();
		}

	}
	
	
}
