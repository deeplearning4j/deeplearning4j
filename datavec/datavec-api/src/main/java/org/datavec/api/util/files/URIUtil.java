package org.datavec.api.util.files;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * Lightweight utilities for converting files to URI.
 *
 * @author Justin Long (crockpotveggies)
 */
public class URIUtil {

    public static URI fileToURI(File f) {
        try {
            // manually construct URI (this is faster)
            String sp = slashify(f.getAbsoluteFile().getPath(), false);
            if (!sp.startsWith("//"))
                sp = "//" + sp;
            return new URI("file", null, sp, null);

        } catch (URISyntaxException x) {
            throw new Error(x); // Can't happen
        }
    }

    private static String slashify(String path, boolean isDirectory) {
        String p = path;
        if (File.separatorChar != '/')
            p = p.replace(File.separatorChar, '/');
        if (!p.startsWith("/"))
            p = "/" + p;
        if (!p.endsWith("/") && isDirectory)
            p = p + "/";
        return p;
    }
}
