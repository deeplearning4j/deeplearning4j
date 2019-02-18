package org.datavec.python;

import org.nd4j.linalg.io.ClassPathResource;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class RestrictedPython {

    private static String escapeStr(String str){
        str = str.replace("\\", "\\\\");
        str = str.replace("\"\"\"", "\\\"\\\"\\\"");
        return str;
    }
    private static String readTXT(String file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        StringBuilder stringBuilder = new StringBuilder();
        char[] buffer = new char[10];
        while (reader.read(buffer) != -1) {
            stringBuilder.append(new String(buffer));
            buffer = new char[10];
        }
        reader.close();
        return stringBuilder.toString();
    }


    public static String getSafeCode(String code) throws IOException{
        String safeCode =readTXT(new ClassPathResource("restricted_python.py").getFilename());
        safeCode = safeCode.replace("<user-code>", escapeStr(code));
        return safeCode;
    }

}
