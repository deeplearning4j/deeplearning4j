package com.ccc.deeplearning.util;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;

public class ImageLoader {


	
	
	public DoubleMatrix asRowVector(File f) throws Exception {
		return MatrixUtil.toMatrix(flattenedImageFromFile(f));
	}
	
	public DoubleMatrix asMatrix(File f) throws IOException {
		return MatrixUtil.toMatrix(fromFile(f));
	}
	
	public int[] flattenedImageFromFile(File f) throws Exception {
		return ArrayUtil.flatten(fromFile(f));
	}
	
	public int[][] fromFile(File file) throws IOException  {
		BufferedImage image = ImageIO.read(file);
	    Raster raster = image.getData();
	    int w = raster.getWidth(),h = raster.getHeight();
        int[][] ret = new int[w][h];
		for(int i = 0; i < w; i++)
			for(int j = 0; j < h; j++)
				ret[i][j] = raster.getSample(i, j, 0);
        
		return ret;
	}

}
