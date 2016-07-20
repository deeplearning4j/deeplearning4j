package org.nd4j.nativeblas;


import java.util.Properties;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Platform;

/**
 * Created by agibsonccc on 2/20/16.
 */
@Platform(include = "NativeLapack.h", compiler = "cpp11", link = "nd4j", library = "jnind4j")
public class NativeLapack extends Pointer {
    static {
        // using our custom platform properties from resources, and on user request,
        // load in priority libraries found in the library path over bundled ones
        String platform = Loader.getPlatform();
        Properties properties = Loader.loadProperties(platform + "-nd4j", platform);
        properties.remove("platform.preloadpath");
        String s = System.getProperty("org.nd4j.nativeblas.pathsfirst", "false").toLowerCase();
        boolean pathsFirst = s.equals("true") || s.equals("t") || s.equals("");
        try {
            Loader.load(NativeLapack.class, properties, pathsFirst);
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("ND4J is probably missing dependencies. For more information, please refer to: http://nd4j.org/getstarted.html", e);
        }
    }

    public NativeLapack() {
    }
// LU decomoposition of a general matrix

    /**
     * LU decomposiiton of a matrix
     * @param M
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param INFO
     */
    public native void dgetrf(long[] extraPointers,int M, int N, long A, int lda, int[] IPIV, int INFO);

    // generate inverse of a matrix given its LU decomposition

    /**
     * Generate inverse ggiven LU decomp
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    public native void dgetri(long[] extraPointers,int N, long A, int lda, int[] IPIV, long WORK, int lwork, int INFO);

    // LU decomoposition of a general matrix

    /**
     * LU decomposiiton of a matrix
     * @param M
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param INFO
     */
    public native void sgetrf(long[] extraPointers,int M, int N, long A, int lda, int[] IPIV, int INFO);

    // generate inverse of a matrix given its LU decomposition

    /**
     * Generate inverse ggiven LU decomp
     * @param N
     * @param A
     * @param lda
     * @param IPIV
     * @param WORK
     * @param lwork
     * @param INFO
     */
    public native void sgetri(long[] extraPointers,int N, long A, int lda, int[] IPIV, long WORK, int lwork, int INFO);
}
