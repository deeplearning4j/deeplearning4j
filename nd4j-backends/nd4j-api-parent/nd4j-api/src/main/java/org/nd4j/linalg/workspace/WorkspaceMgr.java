package org.nd4j.linalg.workspace;

import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * WorkspaceMgr is an interface for managing a set of workspaces, for a set of array types (where the array types
 * are specified by an enumeration).
 * Note that multiple array types may be stored in the one underlying workspace
 *
 * @param <T> Enumeration type to specify the type of array. For example, in DL4J the type values include things
 *            like inputs, activations, working memory etc.
 * @author Alex Black
 */
public interface WorkspaceMgr<T extends Enum<T>> {

    /**
     * Set the workspace name for the specified array type
     *
     * @param arrayType Array type to set the workspace name for
     * @param wsName    Workspace name to set
     */
    void setWorkspaceName(T arrayType, String wsName);

    /**
     * @param arrayType Array type to get the workspace name for (if set)
     * @return The workspace name for the specified array type (or null, if none has been set)
     */
    String getWorkspaceName(T arrayType);

    /**
     * Seth the workspace name and configuration for the specified array type
     *
     * @param arrayType     Array type
     * @param wsName        Workspace name
     * @param configuration Workspace configuration
     */
    void setWorkspace(T arrayType, String wsName, WorkspaceConfiguration configuration);

    /**
     * Set the workspace configuration for the specified array type
     *
     * @param arrayType     Type of array to set the configuration for
     * @param configuration Configuration for the specified array type
     */
    void setConfiguration(T arrayType, WorkspaceConfiguration configuration);

    /**
     * @param arrayType Array type to get the workspace configuration for
     * @return Workspace configuration for the specified array type (or note, if no configuration has been set)
     */
    WorkspaceConfiguration getConfiguration(T arrayType);

    /**
     * Set arrays to be scoped out (not in any workspace) for the specified array type.
     * This means that create, dup, leverage etc methods will return result arrays that are not attached to any workspace
     *
     * @param arrayType Array type to set scoped out for
     */
    void setScopedOutFor(T arrayType);

    /**
     * @param arrayType Array type
     * @return True if the specified array type is set to be scoped out
     */
    boolean isScopedOut(T arrayType);

    /**
     * Has the specified array type been configured in this workspace manager?
     *
     * @param arrayType Array type to check
     * @return True if the array type has been configured (either scoped out, or a workspace has been set for this
     *  array type)
     */
    boolean hasConfiguration(T arrayType);

    /**
     * @param arrayType Array type to enter the scope for
     * @return Workspace for the specified array type
     */
    MemoryWorkspace notifyScopeEntered(T arrayType);

    /**
     * Open/enter multiple workspaces. This is equivalent to nested opening of the specified workspaces
     *
     * @param arrayTypes Open the specified workspaces
     * @return Closeable for the specified workspaces
     */
    WorkspacesCloseable notifyScopeEntered(T... arrayTypes);

    /**
     * Borrow the scope for the specified array type
     *
     * @param arrayType Array type to borrow the scope for
     * @return Workspace
     */
    MemoryWorkspace notifyScopeBorrowed(T arrayType);

    /**
     * Check if the workspace for the specified array type is open. If the array type is set to be scoped out,
     * this will return true
     *
     * @param arrayType Array type
     * @return True if the workspace is open (or array type is set to scoped out)
     */
    boolean isWorkspaceOpen(T arrayType);

    /**
     * Assert thath the workspace for the specified array type is open.
     * For array types that are set to scoped out, this will be treated as a no-op
     * @param arrayType Array type to check
     * @param msg       May be null. If non-null: include this in the exception
     * @throws ND4JWorkspaceException If the specified workspace is not open
     */
    void assertOpen(T arrayType, String msg) throws ND4JWorkspaceException;

    /**
     * Assert thath the workspace for the specified array type is not open.
     * For array types that are set to scoped out, this will be treated as a no-op
     * @param arrayType Array type to check
     * @param msg       May be null. If non-null: include this in the exception
     * @throws ND4JWorkspaceException If the specified workspace is open
     */
    void assertNotOpen(T arrayType, String msg) throws ND4JWorkspaceException;

    /**
     * Assert that the current workspace is the one for the specified array type.
     * As per {@link #isWorkspaceOpen(Enum)} scoped out array types are ignored here.
     *
     * @param arrayType Array type to check
     * @param msg       May be null. Message to include in the exception
     */
    void assertCurrentWorkspace(T arrayType, String msg) throws ND4JWorkspaceException;

    /**
     * Leverage the array to the specified array type's workspace (or detach if required).
     * If the array is not attached (not defined in a workspace) - array is returned unmodified
     *
     * @param toWorkspace Array type's workspace to move the array to
     * @param array       Array to leverage
     * @return Leveraged array (if leveraged, or original array otherwise)
     */
    INDArray leverageTo(T toWorkspace, INDArray array);

    /**
     * Validate that the specified array type is actually in the workspace it's supposed to be in
     *
     * @param arrayType           Array type of the array
     * @param array               Array to check
     * @param migrateIfInvalid    if true, and array is in the wrong WS: migrate the array and return. If false and in
     *                            the wrong WS: exception
     * @param exceptionIfDetached If true: if the workspace is detached, but is expected to be in a workspace: should an
     *                            exception be thrown?
     * @return The original array, or (if required, and if migrateIfInvalid == true) the migrated array
     * @throws ND4JWorkspaceException If the array is in the incorrect workspace (and migrateIfInvalid == false)
     */
    INDArray validateArrayLocation(T arrayType, INDArray array, boolean migrateIfInvalid, boolean exceptionIfDetached) throws ND4JWorkspaceException;

    /**
     * Create an array in the specified array type's workspace (or detached if none is specified).
     * Equivalent to {@link org.nd4j.linalg.factory.Nd4j#create(int...)}, other than the array location
     * @param arrayType Array type
     * @param shape     Shape
     * @return Created arary
     */
    INDArray create(T arrayType, int... shape);

    /**
     * Create an array in the specified array type's workspace (or detached if none is specified).
     * Equivalent to {@link org.nd4j.linalg.factory.Nd4j#create(int[],char)}, other than the array location
     * @param arrayType Array type
     * @param shape     Shape
     * @param ordering Order of the array
     * @return Created arary
     */
    INDArray create(T arrayType, int[] shape, char ordering);

    /**
     * Create an uninitialized array in the specified array type's workspace (or detached if none is specified).
     * Equivalent to {@link org.nd4j.linalg.factory.Nd4j#createUninitialized(int)} (int...)}, other than the array location
     * @param arrayType Array type
     * @param shape     Shape
     * @return Created array
     */
    INDArray createUninitialized(T arrayType, int... shape);

    /**
     * Create an uninitialized array in the specified array type's workspace (or detached if none is specified).
     * Equivalent to {@link org.nd4j.linalg.factory.Nd4j#createUninitialized(int[], char)}}, other than the array location
     * @param arrayType Array type
     * @param shape     Shape
     * @param order Order of the array
     * @return Created array
     */
    INDArray createUninitialized(T arrayType, int[] shape, char order);

    /**
     * Duplicate the array, where the array is put into the specified array type's workspace (if applicable)
     * @param arrayType Array type for the result
     * @param toDup     Array to duplicate
     * @return Duplicated array in the specified array type's workspace
     */
    INDArray dup(@NonNull T arrayType, @NonNull INDArray toDup);

    /**
     * Duplicate the array, where the array is put into the specified array type's workspace (if applicable)
     * @param arrayType Array type for the result
     * @param toDup     Array to duplicate
     * @param order     Order for the duplicated array
     * @return Duplicated array in the specified array type's workspace
     */
    INDArray dup(T arrayType, INDArray toDup, char order);


}
