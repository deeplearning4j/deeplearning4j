package org.deeplearning4j.text.treeparser;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.StringUtils;
import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.transformer.TreeTransformer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Binarizes trees.
 * Based on the work by Manning et. al in stanford corenlp
 *
 * @author Adam Gibson
 */
public class BinarizeTreeTransformer implements TreeTransformer {

    private HeadWordFinder headWordFinder = new HeadWordFinder();
    private boolean markovFactor = false;
    private boolean insideFactor = false;
    private boolean simpleLabels = false;
    private boolean markFinalStates = false;
    private boolean unaryAtTop = false;
    private boolean doSelectiveSplit = false;
    private boolean useWrappingLabels = false;
    private boolean noRebinarization = false;
    private int selectiveSplitThreshold,markovOrder;
    private Counter<String> stateCounter;
    private static Logger log = LoggerFactory.getLogger(BinarizeTreeTransformer.class);

    @Override
    public Tree transform(Tree t) {
        if(t == null)
            return null;


        String tLabelVal = t.label();
        if(t.isLeaf()) {
            Tree ret = new Tree(t);
            ret.setLabel(tLabelVal);
            return ret;
        }

         if(t.isPreTerminal()) {
            Tree child = transform(t.firstChild());
            String val = child.value();
            List<Tree> newChildren = new ArrayList<>(1);

            Tree add = new Tree(child.getTokens());
            add.setLabel(child.label());
            add.setTags(Arrays.asList(tLabelVal));
            add.setType(child.getType());
            add.setValue(val);
            newChildren.add(add);
            child.connect(newChildren);
            return child;

        }


        Tree head = headWordFinder.findHead(t);

        if(head == null && !t.label().equals("S") || !t.label().equals("TOP"))
            log.warn("No head found");

        int headNum = -1;
        List<Tree> children = head.children();
        List<Tree> newChildren = new ArrayList<>();
        for(int i = 0; i < children.size(); i++) {
            Tree child = children.get(i);
            Tree childResult = transform(child);
            if(childResult == head)
                headNum = i;
            newChildren.add(child);
        }


        Tree result;
        if(t.label().charAt(0) == 'S' || t.label().equals("TOP")) {
            result = new Tree(t);
            result.connect(newChildren);
        }

        else {
            String word = head.value();
            String tag = head.tags().get(0);
            result = new Tree(t.getTokens());
            result.connect(newChildren);
            result.setTags(Arrays.asList(tag));
            result.setLabel(tLabelVal + "[" + word + "/" + tag + "]");
            result.setValue(word);
            result = binarizeLocalTree(t,headNum,word,tag);
        }

        return result;

    }

    Tree binarizeLocalTree(Tree t, int headNum, String word,String tag) {
        if (markovFactor) {
            String topCat = t.label();
            Tree t2;
            if (insideFactor) {
                t2 = markovInsideBinarizeLocalTreeNew(t, headNum, 0, t.children().size() - 1, true);
            } else {
                t2 = markovOutsideBinarizeLocalTree(t, word,tag, headNum, topCat, new LinkedList<Tree>(), false);
            }

            return t2;
        }
        if (insideFactor) {
            return insideBinarizeLocalTree(t, headNum, word,tag, 0, 0);
        }
        return outsideBinarizeLocalTree(t, t.label(), t.label(), headNum, word,tag, 0, "", 0, "");
    }

    private Tree markovOutsideBinarizeLocalTree(Tree t, String word,String tag, int headLoc, String topCat, LinkedList<Tree> ll, boolean doneLeft) {

        List<Tree> newChildren = new ArrayList<>(2);
        // call with t, headNum, head, topCat, false
        if (headLoc == 0) {
            if (!doneLeft) {
                // insert a unary to separate the sides
                if (topCat.equals("S") || topCat.equals("TOP")) {
                    return markovOutsideBinarizeLocalTree(t, word,tag, headLoc, topCat, new LinkedList<Tree>(), true);
                }
                String subLabelStr;
                if (simpleLabels) {
                    subLabelStr = "@" + topCat;
                } else {
                    String headStr = t.children().get(headLoc).label();
                    subLabelStr = "@" + topCat + ": " + headStr + " ]";
                }

                Tree subTree = new Tree(t);
                subTree.setTags(Collections.singletonList(tag));
                subTree.setLabel(subLabelStr + "[" + word + "/" + tag);
                subTree.connect(t.children());
                newChildren.add(markovOutsideBinarizeLocalTree(subTree, word,tag, headLoc, topCat, new LinkedList<Tree>(), true));
                Tree ret = new Tree(t);
                ret.connect(newChildren);
                return ret;

            }


            int len = t.children().size();
            // len = 1
            if (len == 1) {
                Tree ret = new Tree(t);
                ret.connect(Collections.singletonList(t.children().get(0)));
                return ret;
            }

            ll.addFirst(t.children().get(len - 1));
            if (ll.size() > markovOrder) {
                ll.removeLast();
            }
            // generate a right
            String subLabelStr;
            if (simpleLabels) {
                subLabelStr = "@" + topCat;
            } else {
                String headStr = t.children().get(headLoc).label();
                String rightStr = (len > markovOrder - 1 ? "... " : "") +  StringUtils.join(ll);
                subLabelStr = "@" + topCat + ": " + headStr + " " + rightStr;
            }



            Tree subTree = new Tree(t.getTokens());
            subTree.connect(t.children().subList(0, len - 1));
            subTree.setValue(word);
            subTree.setLabel(tag);
            subTree.setHeadWord(subLabelStr);
            newChildren.add(markovOutsideBinarizeLocalTree(subTree, word,tag, headLoc, topCat, ll, true));
            newChildren.add(t.children().get(len - 1));
            Tree ret = new Tree(t);
            ret.connect(newChildren);
            return ret;
        }
        if (headLoc > 0) {
            ll.add(t.children().get(0));
            if (ll.size() > markovOrder) {
                ll.remove(0);
            }
            // generate a left
            String subLabelStr;
            if (simpleLabels) {
                subLabelStr = "@" + topCat;
            } else {
                String headStr = t.children().get(headLoc).label();
                String leftStr = StringUtils.join(ll) + (headLoc > markovOrder - 1 ? " ..." : "");
                subLabelStr = "@" + topCat + ": " + leftStr + " " + headStr + " ]";
            }

            Tree subTree = new Tree(t.getTokens());
            subTree.setLabel(subLabelStr);
            subTree.setTags(Collections.singletonList(tag));
            subTree.setValue(word);
            subTree.connect(t.children().subList(1, t.children().size()));
            newChildren.add(t.children().get(0));
            newChildren.add(markovOutsideBinarizeLocalTree(subTree, word,tag, headLoc - 1, topCat, ll, false));


            Tree ret = new Tree(t);
            ret.setLabel(t.label());
            ret.connect(newChildren);
            return ret;
        }
        return t;
    }

    /**
     * Uses tail recursion. The Tree t that is passed never changes, only the indices left and right do.
     */
    private Tree markovInsideBinarizeLocalTreeNew(Tree t, int headLoc, int left, int right, boolean starting) {
        Tree result;
        List<Tree> children = t.children();
        if (starting) {
            // this local tree is a unary and doesn't need binarizing so just return it
            if (left == headLoc && right == headLoc) {
                return t;
            }
            // this local tree started off as a binary and the option to not
            // rebinarized such trees is set
            if (noRebinarization && children.size() == 2) {
                return t;
            }
            if (unaryAtTop) {
                // if we're doing grammar compaction, we add the unary at the top
                result = new Tree(t);
                result.connect(Collections.singletonList(markovInsideBinarizeLocalTreeNew(t, headLoc, left, right, false)));
                return result;
            }
        }
        // otherwise, we're going to make a new tree node
        List<Tree> newChildren = null;
        // left then right top down, this means we generate right then left on the way up
        if (left == headLoc && right == headLoc) {
            // base case, we're done, just make a unary
            newChildren = Collections.singletonList(children.get(headLoc));
        } else if (left < headLoc) {
            // generate a left if we can
            newChildren = new ArrayList<>(2);
            newChildren.add(children.get(left));
            newChildren.add(markovInsideBinarizeLocalTreeNew(t, headLoc, left + 1, right, false));
        } else if (right > headLoc) {
            // generate a right if we can
            newChildren = new ArrayList<>(2);
            newChildren.add(markovInsideBinarizeLocalTreeNew(t, headLoc, left, right - 1, false));
            newChildren.add(children.get(right));
        } else {
            // this shouldn't happen, should have been caught above
            log.warn("Bad bad parameters passed to markovInsideBinarizeLocalTree");
        }
        // newChildren should be set up now with two children
        // make our new label
        String label;
        if (starting) {
            label = t.label();
        } else {
            label = makeSyntheticLabel(t, left, right, headLoc, markovOrder);
        }
        if (doSelectiveSplit) {
            double stateCount = stateCounter.getCount(label);
            if (stateCount < selectiveSplitThreshold) { // too sparse, so
                if (starting && !unaryAtTop) {
                    // if we're not compacting grammar, this is how we make sure the top state has the passive symbol
                    label = t.label();
                } else {
                    label = makeSyntheticLabel(t, left, right, headLoc, markovOrder - 1); // lower order
                }
            }
        } else {
            // otherwise, count up the states
            stateCounter.incrementCount(label, 1.0); // we only care about the category
        }

        // finished making new label
        result = new Tree(t);
        result.setLabel(label);
        result.connect(newChildren);
        return result;
    }


    private String makeSyntheticLabel(Tree t, int left, int right, int headLoc, int markovOrder) {
        String result;
        if (simpleLabels) {
            result = makeSimpleSyntheticLabel(t);
        } else if (useWrappingLabels) {
            result = makeSyntheticLabel2(t, left, right, headLoc, markovOrder);
        } else {
            result = makeSyntheticLabel1(t, left, right, headLoc, markovOrder);
        }
        return result;
    }

    /**
     * Do nothing other than decorate the label with @
     */
    private static String makeSimpleSyntheticLabel(Tree t) {
        String topCat = t.label();
        String labelStr = "@" + topCat;
        String word = t.value();
        String tag = t.tags() != null ? t.tags().get(0) : "";
        return topCat + labelStr + word + tag;
    }

    /**
     * For a dotted rule VP^S -> RB VP NP PP . where VP is the head
     * makes label of the form: @VP^S| [ RB [VP] ... PP ]
     * where the constituent after the @ is the passive that we are building
     * and  the constituent in brackets is the head
     * and the brackets on the left and right indicate whether or not there
     * are more constituents to add on those sides.
     */
    private static String makeSyntheticLabel1(Tree t, int left, int right, int headLoc, int markovOrder) {
        String topCat = t.label();
        List<Tree> children = t.children();
        String leftString;
        if (left == 0) {
            leftString = "[ ";
        } else {
            leftString = " ";
        }
        String rightString;
        if (right == children.size() - 1) {
            rightString = " ]";
        } else {
            rightString = " ";
        }
        for (int i = 0; i < markovOrder; i++) {
            if (left < headLoc) {
                leftString = leftString + children.get(left).label() + " ";
                left++;
            } else if (right > headLoc) {
                rightString = " " + children.get(right).label() + rightString;
                right--;
            } else {
                break;
            }
        }
        if (right > headLoc) {
            rightString = "..." + rightString;
        }
        if (left < headLoc) {
            leftString = leftString + "...";
        }
        String labelStr = "@" + topCat + "| " + leftString + "[" + t.children().get(headLoc).label() + "]" + rightString; // the head in brackets
        String word = t.value();
        String tag = t.tags().get(0);
        return labelStr + "/" + word + "/" + tag;
    }

    /**
     * for a dotted rule VP^S -> RB VP NP PP . where VP is the head
     * makes label of the form: @VP^S| VP_ ... PP> RB[
     */
    private String makeSyntheticLabel2(Tree t, int left, int right, int headLoc, int markovOrder) {
        String topCat = t.label();
        List<Tree> children = t.children();
        String finalPiece;
        int i = 0;
        if (markFinalStates) {
            // figure out which one is final
            if (headLoc != 0 && left == 0) {
                // we are finishing on the left
                finalPiece = " " + children.get(left).label() + "[";
                left++;
                i++;
            } else if (headLoc == 0 && right > headLoc && right == children.size() - 1) {
                // we are finishing on the right
                finalPiece = " " + children.get(right).label() + "]";
                right--;
                i++;
            } else {
                finalPiece = "";
            }
        } else {
            finalPiece = "";
        }

        String middlePiece = "";
        for (; i < markovOrder; i++) {
            if (left < headLoc) {
                middlePiece = " " + children.get(left).label() + "<" + middlePiece;
                left++;
            } else if (right > headLoc) {
                middlePiece = " " + children.get(right).label() + ">" + middlePiece;
                right--;
            } else {
                break;
            }
        }
        if (right > headLoc || left < headLoc) {
            middlePiece = " ..." + middlePiece;
        }


        String headStr = t.children().get(headLoc).label();
        // Optimize memory allocation for this next line, since these are the
        // String's that linger.
        // String labelStr = "@" + topCat + "| " + headStr + "_" + middlePiece + finalPiece;
        int leng = 1 + 2 + 1 + topCat.length() + headStr.length() + middlePiece.length() + finalPiece.length();
        StringBuilder sb = new StringBuilder(leng);
        sb.append("@").append(topCat).append("| ").append(headStr).append("_").append(middlePiece).append(finalPiece);
        String labelStr = sb.toString();
        // System.err.println("makeSyntheticLabel2: " + labelStr);

        String word = t.value();
        String tag = t.tags().get(0);
        return labelStr + "/[" + word + "/"+ tag + "]";
    }

    private Tree insideBinarizeLocalTree(Tree t, int headNum, String word,String tag, int leftProcessed, int rightProcessed) {
        List<Tree> newChildren = new ArrayList<>(2);      // check done
        if (t.children().size() <= leftProcessed + rightProcessed + 2) {
            Tree leftChild = t.children().get(leftProcessed);
            newChildren.add(leftChild);
            if (t.children().size() == leftProcessed + rightProcessed + 1) {
                // unary ... so top level
                String finalCat = t.label();
                Tree ret = new Tree(t);
                ret.setTags(Collections.singletonList(tag));
                ret.setValue(word);
                ret.setHeadWord(finalCat);
                ret.connect(newChildren);
                return ret;
            }
            // binary
            Tree rightChild = t.children().get(leftProcessed + 1);
            newChildren.add(rightChild);
            String labelStr = t.label();
            if (leftProcessed != 0 || rightProcessed != 0) {
                labelStr = ("@ " + leftChild.label() + " " + rightChild.label());
            }

            Tree ret = new Tree(t);
            ret.setLabel(labelStr);
            ret.setValue(word);
            ret.setTags(Collections.singletonList(tag));
            ret.connect(newChildren);
            return ret;
        }
        if (headNum > leftProcessed) {
            // eat left word
            Tree leftChild = t.children().get(leftProcessed);
            Tree rightChild = insideBinarizeLocalTree(t, headNum, word,tag, leftProcessed + 1, rightProcessed);
            newChildren.add(leftChild);
            newChildren.add(rightChild);
            String labelStr = ("@ " + leftChild.label() + " " + rightChild.label().substring(2));
            if (leftProcessed == 0 && rightProcessed == 0) {
                labelStr = t.label();
            }

            Tree ret = new Tree(t);
            ret.connect(newChildren);
            ret.setHeadWord(tag);
            ret.setLabel(labelStr);
            ret.setValue(word);

            return ret;
        } else {
            // eat right word
            Tree leftChild = insideBinarizeLocalTree(t, headNum, word,tag, leftProcessed, rightProcessed + 1);
            Tree rightChild = t.children().get(t.children().size() - rightProcessed - 1);
            newChildren.add(leftChild);
            newChildren.add(rightChild);
            String labelStr = ("@ " + leftChild.label().substring(2) + " " + rightChild.label());
            if (leftProcessed == 0 && rightProcessed == 0) {
                labelStr = t.label();
            }

            Tree ret = new Tree(t);
            ret.setLabel(labelStr);
            ret.setValue(word);
            ret.connect(newChildren);
            return ret;
        }
    }

    private Tree outsideBinarizeLocalTree(Tree t, String labelStr, String finalCat, int headNum, String word,String tag, int leftProcessed, String leftStr, int rightProcessed, String rightStr) {
        List<Tree> newChildren = new ArrayList<>(2);
        String label = labelStr + "/" + word + "/tag";
        // check if there are <=2 children already
        if (t.children().size() - leftProcessed - rightProcessed <= 2) {
            // done, return
            newChildren.add(t.children().get(leftProcessed));
            if (t.children().size() - leftProcessed - rightProcessed == 2) {
                newChildren.add(t.children().get(leftProcessed + 1));
            }

            Tree ret = new Tree(t);
            ret.connect(newChildren);
            ret.setLabel(label);

            return ret;
        }
        if (headNum > leftProcessed) {
            // eat a left word
            Tree leftChild = t.children().get(leftProcessed);
            String childLeftStr = leftStr + " " + leftChild.label();
            String childLabelStr;
            if (simpleLabels) {
                childLabelStr = "@" + finalCat;
            } else {
                childLabelStr = "@" + finalCat + " :" + childLeftStr + " ..." + rightStr;
            }
            Tree rightChild = outsideBinarizeLocalTree(t, childLabelStr, finalCat, headNum, word,tag, leftProcessed + 1, childLeftStr, rightProcessed, rightStr);
            newChildren.add(leftChild);
            newChildren.add(rightChild);

            Tree ret = new Tree(t);
            ret.connect(newChildren);
            ret.setLabel(label);

            return ret;
        } else {
            // eat a right word
            Tree rightChild = t.children().get(t.children().size() - rightProcessed - 1);
            String childRightStr = " " + rightChild.label() + rightStr;
            String childLabelStr;
            if (simpleLabels) {
                childLabelStr = "@" + finalCat;
            } else {
                childLabelStr = "@" + finalCat + " :" + leftStr + " ..." + childRightStr;
            }

            Tree leftChild = outsideBinarizeLocalTree(t, childLabelStr, finalCat, headNum, word,tag, leftProcessed, leftStr, rightProcessed + 1, childRightStr);
            newChildren.add(leftChild);
            newChildren.add(rightChild);
            Tree ret = new Tree(t);
            ret.connect(newChildren);
            ret.setLabel(labelStr);

            return ret;
        }
    }





}
