package nilgiri.processing;


public class ViewConfig2D {
	private static final int DIM = 2;
	private static final int X = 0;
	private static final int Y = 1;
	private float m_translation[] = {0.0f, 0.0f};
	private float m_scale = 1.0f; 
//	private float m_rotation = 0.0f;	

	public ViewConfig2D(){
	}

	public void setTranslation(float[] i_v){
		System.arraycopy(i_v, 0, m_translation, 0, DIM);
	}
	public void setTranslation(float i_x, float i_y){ 
		m_translation[X] = i_x;
		m_translation[Y] = i_y;
	}

	public void setScale(float i_scale){
		m_scale = i_scale;
	}
	

	public void getTranslation(float[] i_v){
		System.arraycopy(m_translation, 0, i_v, 0, DIM);
	}

	public float getScale(){
		return m_scale;
	}
}
