package org.deeplearning4j.datasets;


import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;


public class MNISTViewer {

	MnistManager manager;
	NN network;
	InputPanel input;
	OutputPanel output;
	OptionsPanel options;
	
	int xnodes = 50;
	int ynodes = 24;
	int connections = 10;
	
	int outputRows = 10;
	int outputCols = 10;

	public MNISTViewer(){

		// Load the data manager
		try {
			manager = new MnistManager("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		
		network = new NN(xnodes*ynodes, connections);
		network.init();
		
		// Setup viewer
		MyFrame mf = new MyFrame("MNSIT Viewer");

		//Display the viewer.
		mf.pack();
		mf.setVisible(true);		
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		MNISTViewer viewer = new MNISTViewer();
	}

	class MyFrame extends JFrame {

		public MyFrame(String s){
			super(s);
			setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			JPanel main = new JPanel();
			main.setBorder(BorderFactory.createLineBorder(Color.black));
			main.setLayout(new GridLayout(1, 3));

			input = new InputPanel();
			output = new OutputPanel();
			options = new OptionsPanel();

			add(main);
			main.add(input); 
			main.add(output); 
			main.add(options);

		}
	}	

	class InputPanel extends JPanel{

		private int width = 200;
		private int height = 200;
		private int mx = 20; // Margin x
		private int my = 30; // Margin y	
		
		private int imageIndex = 1;
		private int maxIndex;

		public InputPanel(){
			super();
			setPreferredSize(new Dimension(width, height));
			setBorder(BorderFactory.createLineBorder(Color.black));
			JLabel title = new JLabel("Input");
			add(title);	
			manager.setCurrent(imageIndex);
			maxIndex = manager.getImages().getCount();
		}
		
		public void nextImage(){
			imageIndex = imageIndex + 1 > maxIndex ? maxIndex : imageIndex + 1;
			manager.setCurrent(imageIndex);
		}
		
		public void previousImage(){
			imageIndex = imageIndex - 1 < 1 ? 1 : imageIndex - 1;
			manager.setCurrent(imageIndex);
		}	
		
		public void drawCurrentImage(Graphics g){
			manager.setCurrent(imageIndex);
			int[][] image = null;
			int rows = manager.getImages().getRows();
			int cols = manager.getImages().getRows();
			
			try {
				image = manager.readImage();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			for (int i = 0; i < rows; i++){
				for (int j = 0; j < cols; j++){
					int c = image[i][j];
					g.setColor(new Color(c, c, c));
					g.fillRect(mx+j, my + i, 1, 1);
				}
				//System.out.println();
			}
		}
		
		public void drawCurrentOutput(Graphics g){
	
			int xoffset = mx;
			int yoffset = 3 * my;			
			
			int target;
			try {
				target = manager.readLabel();
				int[][] values = new int[outputRows][outputCols];
				for (int i = 0; i < outputCols; i++){
					values[target][i] = 255;;
				}
				int c = 0;
				for (int i = 0; i < outputRows; i++){
					for (int j = 0; j < outputCols; j++){
						c = values[i][j];
						g.setColor(new Color(c, c, c));
						g.fillRect(xoffset+j, yoffset + i, 1, 1);
					}
				}				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}		

		@Override
		public void paintComponent(Graphics g){
			super.paintComponent(g);
			g.drawRect(mx, my, getWidth()-2*mx, getHeight()-2*my);
			drawCurrentImage(g);
			drawCurrentOutput(g);
		}

	}	

	class OutputPanel extends JPanel{

		private int width = 200;
		private int height = 200;
		private int mx = 20; // Margin x
		private int my = 30; // Margin y

		public OutputPanel(){
			super();
			setPreferredSize(new Dimension(width, height));
			setBorder(BorderFactory.createLineBorder(Color.black));
			JLabel title = new JLabel("Output");
			add(title);
		}

		private void drawInputNodes(Graphics g){
			
			int rows = manager.getImages().getRows();
			int cols = manager.getImages().getRows();
			
			float[] state = network.readInput();
			if (state == null){
				state = new float[rows*cols];
			}
			
			int xoffset = mx;
			int yoffset = my;
			
			for (int i = 0; i < rows; i++){
				for (int j = 0; j < cols; j++){
					int index = i * cols + j;
					int v = (int)(255.0 * state[index]);
					int c = v > 255 ? 255 : v;
					try {
					g.setColor(new Color(c, c, c));
					g.fillRect(xoffset+j, yoffset + i, 1, 1);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}				
		}		
		
		private void drawState(Graphics g){
			float[] state = network.getState();
			
			int xoffset = mx;
			int yoffset = 3 * my;
			
			int rows = ynodes;
			int cols = xnodes;
			
			for (int i = 0; i < rows; i++){
				for (int j = 0; j < cols; j++){
					int v = (int)(255.0 * state[i * cols + j]);
					int c = v > 255 ? 255 : v;					
					g.setColor(new Color(c, c, c));
					g.fillRect(xoffset+j, yoffset + i, 1, 1);
				}
			}				
		}
		
		private void drawOutputNodes(Graphics g){
			
			int rows = outputRows;
			int cols = outputCols;
			
			float[] state = network.readOutput();
			if (state == null){
				state = new float[rows*cols];
			}
			
			int xoffset =  mx;
			int yoffset = 4 * my;
			
			for (int i = 0; i < rows; i++){
				for (int j = 0; j < cols; j++){
					int index = i * cols + j;
					int v = (int)(255.0 * state[index]);
					int c = v > 255 ? 255 : v;
					try {
					g.setColor(new Color(c, c, c));
					g.fillRect(xoffset+j, yoffset + i, 1, 1);
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}				
		}			

		@Override
		public void paintComponent(Graphics g){
			super.paintComponent(g);
			g.drawRect(mx, my, getWidth()-2*mx, getHeight()-2*my);
			drawInputNodes(g);
			drawState(g);
			drawOutputNodes(g);
		}	

	}

	class OptionsPanel extends JPanel{

		public OptionsPanel(){
			super();
			setPreferredSize(new Dimension(200,200));
			setBorder(BorderFactory.createLineBorder(Color.black));
			JLabel title = new JLabel("Options");
			add(title);
			JButton next = new JButton("Next");
			JButton previous = new JButton("Previous");
			JButton setInput = new JButton("Set Input");
			JButton setOutput = new JButton("Set Output");
			JButton update = new JButton("Update");
			JButton reset = new JButton("Reset");
			
			next.addActionListener
			(
				new ActionListener(){
					public void actionPerformed( ActionEvent e )
					{
						input.nextImage();
						input.repaint();
					}
				}
			);
			
			previous.addActionListener(
				new ActionListener(){
					public void actionPerformed( ActionEvent e )
					{
						input.previousImage();
						input.repaint();
					}
				}
			);
			
			setInput.addActionListener(
					new ActionListener(){
						public void actionPerformed( ActionEvent e )
						{
							try {
								// serialise image
								manager.setCurrent(input.imageIndex);
								int[][] image = manager.readImage();
								int size = image.length*image[0].length;
								int[] inputNodes = new int[size];
								for (int i = 0; i < size; i++){
									inputNodes[i] = i;
								}
								float[] inputValues = new float[size];
								for (int i = 0; i < image.length; i++){
									for (int j = 0; j < image[0].length; j++){
										inputValues[i*image.length + j] = (float)(image[i][j])/255f;
									}
								}
								network.setInput(inputNodes, inputValues);
							} catch (IOException e1) {
								// TODO Auto-generated catch block
								e1.printStackTrace();
							}
							
							output.repaint();
						}			
					}
				);		
			
			setOutput.addActionListener(
					new ActionListener(){
						public void actionPerformed( ActionEvent e )
						{

							int size = outputRows*outputCols;
							
							// set output as the last nodes in the network list
							int[] outputNodes = new int[size];
							int index = xnodes*ynodes-1-size;
							for (int i = 0; i < size; i++){
								outputNodes[i] = index;
								index++;
							}
								
							// map output values to current image label
							int target;
							try {
								target = manager.readLabel();
								float[][] values = new float[outputRows][outputCols];
								for (int i = 0; i < outputCols; i++){
									values[target][i] = 1.0f;
								}
								float[] outputValues = new float[size];
								for (int i = 0; i < outputRows; i++){
									for (int j = 0; j < outputCols; j++){
										outputValues[i*outputCols + j] = values[i][j];
									}
								}							

								network.setOutput(outputNodes, outputValues);
								input.repaint();
								output.repaint();								
							} catch (IOException e1) {
								// TODO Auto-generated catch block
								e1.printStackTrace();
							}
						}			
					}
				);				
			
			update.addActionListener(
				new ActionListener(){
					public void actionPerformed( ActionEvent e )
					{
						network.update();
						output.repaint();	
						//System.out.println(network.toString());
					}
				}
			);	
		
			reset.addActionListener(
					new ActionListener(){
						public void actionPerformed( ActionEvent e )
						{
							network.reset();
							output.repaint();	
						}
					}
				);				
			
			add(next); add(previous); add(setInput); add(setOutput); add(update); add(reset);
			
		}

	}

}





