package com.ccc.deeplearning.util;

import java.io.DataInputStream;
import java.io.IOException;

public class ByteUtil {
	/**
	 *
	 * 
	 * @param dis
	 * @return
	 * @throws IOException
	 */
	public static String readString(DataInputStream dis,int maxSize) throws IOException {
		byte[] bytes = new byte[maxSize];
		byte b = dis.readByte();
		int i = -1;
		StringBuilder sb = new StringBuilder();
		while (b != 32 && b != 10) {
			i++;
			bytes[i] = b;
			b = dis.readByte();
			if (i == 49) {
				sb.append(new String(bytes));
				i = -1;
				bytes = new byte[maxSize];
			}
		}
		sb.append(new String(bytes, 0, i + 1));
		return sb.toString();
	}

}
