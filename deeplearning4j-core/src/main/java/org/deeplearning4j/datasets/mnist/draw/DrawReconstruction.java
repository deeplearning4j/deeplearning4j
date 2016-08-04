/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets.mnist.draw;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;


public class DrawReconstruction {

	public  JFrame frame;
	BufferedImage img;
    private INDArray data;
	private int width = 28;
	private int height = 28;
	public String title = "TEST";
	private int heightOffset = 0;
	private int widthOffset = 0;

	
	public DrawReconstruction(INDArray data, int heightOffset, int widthOffset) {
		img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	    this.data = data;
        this.heightOffset = heightOffset;
		this.widthOffset = widthOffset;


	}
	
	public DrawReconstruction(INDArray data) {
		img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		this.data = Transforms.round(data);


	}

	public void readjustToData() {
        this.width = data.columns();
        this.height = data.rows();
        img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

    }


	public void draw() {
        WritableRaster r = img.getRaster();
        int[] equiv = new int[data.length()];
        INDArray dataLinear = data.linearView();
        for(int i = 0; i < equiv.length; i++)
            equiv[i] = Math.round(dataLinear.getInt(i));

        r.setDataElements(0, 0, width, height, equiv);



        frame = new JFrame(title);
		frame.setVisible(true);
		start();
		frame.add(new JLabel(new ImageIcon(getImage())));

		frame.pack();
		// Better to DISPOSE than EXIT
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	}

	public void close() {
		frame.dispose();
	}
	
	public Image getImage() {
		return img;
	}

	public void start(){


		int[] pixels = ((DataBufferInt)img.getRaster().getDataBuffer()).getData();
		boolean running = true;
		while(running){
			BufferStrategy bs = frame.getBufferStrategy();
			if(bs==null){
				frame.createBufferStrategy(4);
				return;
			}
			for (int i = 0; i < width * height; i++)
				pixels[i] = 0;

			Graphics g= bs.getDrawGraphics();
			g.drawImage(img, heightOffset, widthOffset, width, height, null);
			g.dispose();
			bs.show();

		}
	}
}
