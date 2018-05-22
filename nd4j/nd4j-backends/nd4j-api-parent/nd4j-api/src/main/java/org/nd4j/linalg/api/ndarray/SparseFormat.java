package org.nd4j.linalg.api.ndarray;

/**
 * @author Audrey Loeffel
 * <ul>
 *     <li>CSR: Compressed Sparse Row</li>
 *     <li>CSC: Commpressed Sparse Column</li>
 *     <li>COO: Coordinate Matrix Storage</li>
 *     <li>None: No sparse format</li>
 * </ul>
 * @see @see <a href="https://software.intel.com/en-us/node/471374">Sparse Matrix Storage Formats</a>
 */
public enum SparseFormat {
    CSR, COO, NONE
}
