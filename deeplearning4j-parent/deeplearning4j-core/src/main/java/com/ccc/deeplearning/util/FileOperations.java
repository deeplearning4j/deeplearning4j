package com.ccc.deeplearning.util;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class FileOperations {

	private FileOperations() {}
	
	
	
	public static OutputStream createAppendingOutputStream(File to) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to,true));
			return bos;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static void appendTo(String data,File append) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(append,true));
			bos.write(data.getBytes());
			bos.flush();
			bos.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}

}
