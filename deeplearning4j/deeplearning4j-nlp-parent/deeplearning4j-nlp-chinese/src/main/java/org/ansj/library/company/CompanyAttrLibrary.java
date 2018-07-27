package org.ansj.library.company;

import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * 机构名识别词典加载类
 * 
 * @author ansj
 * 
 */
public class CompanyAttrLibrary {

    private static final Log logger = LogFactory.getLog();

    private static HashMap<String, int[]> cnMap = null;

    private CompanyAttrLibrary() {}

    public static HashMap<String, int[]> getCompanyMap() {
        if (cnMap != null) {
            return cnMap;
        }
        init();
        return cnMap;
    }

    // company_freq

    private static void init() {
        try (BufferedReader br = MyStaticValue.getCompanReader()) {
            cnMap = new HashMap<>();
            String temp = null;
            String[] strs = null;
            int[] cna = null;
            while ((temp = br.readLine()) != null) {
                strs = temp.split("\t");
                cna = new int[2];
                cna[0] = Integer.parseInt(strs[1]);
                cna[1] = Integer.parseInt(strs[2]);
                cnMap.put(strs[0], cna);
            }
        } catch (IOException e) {
            logger.warn("IO异常", e);
        }
    }

}
