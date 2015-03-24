/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.util;

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
