package org.nd4j.autodiff.samediff.flow;

import lombok.NonNull;

import java.util.HashMap;
import java.util.Map;

/**
 * This class acts as holder for flow control information.
 *
 * @author raver119@gmail.com
 */
public class FlowPath {
    protected Map<String, NodeState> states = new HashMap<>();
    protected Map<String, FrameState> frames = new HashMap<>();

    /**
     * This method checks if NodeState was created for specified graph node
     *
     * @param nodeName
     */
    public void ensureNodeStateExists(@NonNull String nodeName) {
        if (!states.containsKey(nodeName))
            states.put(nodeName, new NodeState(nodeName));
    }

    /**
     * This method checks, if specified graph node is active (as in - located within active code branch, and was NOT left in inactive branch)
     *
     * @param nodeName
     * @return
     */
    public boolean isActive(@NonNull String nodeName) {
        ensureNodeStateExists(nodeName);

        return states.get(nodeName).isActive();
    }

    /**
     * This method allows to set specified node active or inactive.
     * PLEASE NOTE: All nodes using this node as input, will be considered inactive, if this node is set to be inactive.
     *
     * @param nodeName
     * @param active
     */
    public void markActive(@NonNull String nodeName, boolean active) {
        ensureNodeStateExists(nodeName);

        states.get(nodeName).setActive(active);
    }

    /**
     * This method sets active/inactive branch for divergent nodes (aka Switch)
     *
     * @param nodeName
     * @param branchIdx
     */
    public void setActiveBranch(@NonNull String nodeName, int branchIdx) {
        states.get(nodeName).setActiveBranch(branchIdx);
    }

    /**
     * This method returns active branch of specific node (if any)
     *
     * @param nodeName
     * @return
     */
    public int getActiveBranch(@NonNull String nodeName) {
        return states.get(nodeName).getActiveBranch();
    }

    /**
     * This method returns TRUE if specified node was already executed during current pass, FALSE otherwise
     * @param nodeName
     * @return
     */
    public boolean wasExecuted(@NonNull String nodeName) {
        ensureNodeStateExists(nodeName);

        return states.get(nodeName).isExecuted();
    }

    /**
     * This method allows to toggle wasExecuted() state for specified node
     * @param nodeName
     * @param executed
     */
    public void markExecuted(@NonNull String nodeName, boolean executed) {

        states.get(nodeName).setExecuted(executed);
    }

    /**
     * This node increments number of iterations by 1.
     *
     * @param nodeName
     */
    public void incrementNumberOfCycles(@NonNull String frameName) {
        frames.get(frameName).incrementNumberOfCycles();
    }

    /**
     * This method returns number of iterations of specified node.
     * @param nodeName
     * @return
     */
    public long getNumberOfCycles(@NonNull String frameName) {
        return states.get(frameName).getNumberOfCycles();
    }

    /**
     * This method adds Frame to tracking
     * PLEASE NOTE: Only works for first call, subsequent calls are no-op
     *
     * @param frame_name
     */
    public void registerFrame(@NonNull String frame_name) {
        if (!frames.containsKey(frame_name))
            frames.put(frame_name, new FrameState(frame_name));
    }

    /**
     * This method removes specified frame from tracking
     *
     * @param frame_name
     */
    // FIXME: this approach is probably bad (for backprop) and should be reconsidered
    public void forgetFrame(@NonNull String frame_name) {
        frames.remove(frame_name);
    }

    /**
     * This method returns TRUE if frame_name already registered, false otherwise
     *
     * @param frame_name
     * @return
     */
    public boolean isRegisteredFrame(@NonNull String frame_name) {
        return frames.containsKey(frame_name);
    }

    /**
     * This method checks, if rewind was planned for specified frame_name
     *
     * @return
     */
    public boolean isRewindPlanned(@NonNull String frameName) {
        return frames.get(frameName).isRewindPlanned();
    }


    public boolean isRewindPossible(@NonNull String frameName) {
        return isRewindPlanned(frameName) && getRewindPosition(frameName) >= 0;
    }

    /**
     * This method announces future rewind of graph execution to specified position
     *
     * @param frameName
     */
    public void planRewind(@NonNull String frameName, boolean reallyPlan) {
        frames.get(frameName).setRewindPlanned(reallyPlan);
    }

    /**
     * This method returns planned position within graph for next rewind.
     *
     * @param frameName
     * @return
     */
    public int getRewindPosition(@NonNull String frameName) {
        return frames.get(frameName).getRewindPosition();
    }

    /**
     * This method allows to set position for next rewind within graph
     *
     * @param frameName
     * @param position
     */
    public void setRewindPosition(@NonNull String frameName, int position) {
        frames.get(frameName).setRewindPosition(position);
    }

    /**
     * This method allows to set position for next rewind within graph.
     * PLEASE NOTE: This methods check, if rewind position wasn't set yet. If it was already set for this frame - it'll be no-op method
     *
     * @param frameName
     * @param position
     */
    public void setRewindPositionOnce(@NonNull String frameName, int position) {
        if (getRewindPosition(frameName) >= 0)
            return;

        frames.get(frameName).setRewindPosition(position);
    }

    /**
     * This method triggers frame state
     *
     * @param frameName
     * @param reallyActivate
     */
    public void activateFrame(@NonNull String frameName, boolean reallyActivate) {
        frames.get(frameName).setActive(reallyActivate);
    }

    /**
     * This method returns TRUE if specified frame was activated (as in: Enter/Merge was triggered)
     *
     * @param frameName
     * @return
     */
    public boolean isFrameActive(@NonNull String frameName) {
        return frames.get(frameName).isActive();
    }
}
