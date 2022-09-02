/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

// Created by Adam Gibson 2022 (based on argmax)

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_einsum)

#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/helpers/reductions.h>

namespace sd {
namespace ops {
DECLARE_TYPES(einsum) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})->setAllowedOutputTypes({ALL_INTS});
}
static std::string validString = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
static std::string operators = ",->.";

CUSTOM_OP_IMPL(einsum, -2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (output->isEmpty()) return sd::Status::OK;

  auto axis = *block.getIArguments();

  // axis might be dynamic (i.e. tf mode)
  if (block.width() > 1 && axis.size() == 0) {
    auto axisVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axisVector, axis);
    helpers::argAbsMax(*input, *output, axis);
  } else {
    helpers::argAbsMax(*input, *output, axis);
  }

  STORE_RESULT(output);

  return sd::Status::OK;
}

bool isValidEinSumChar(const char input) {
  return validString.find(input) != std::string::npos
         && operators.find(input) != std::string::npos;
}

char getSymbol(int i) {
  return (char) i;
}

std::vector<char> getUnused(std::string used,int n,std::vector<char> ret) {
  int i,count = 0;
  while(count < n) {
    char currChar = getSymbol(i);
    i += 1;
    if(used.find(currChar) >= 0)
      continue;
    ret.push_back(currChar);
    count += 1;
  }

}


void convertToValidEinSumChars(std::string einsumString,std::string ret) {
  std::set<char> inputSymbols;
  for(char c : einsumString) {
    inputSymbols.insert(c);
  }
  //see: https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/parser.py#L115
  std::set<char> operatorSymbols = {',','-','>'};
  std::set<char> symbols;
  for(char symbol : inputSymbols) {
    if(!operatorSymbols.count(symbol)) {
      symbols.insert(symbol);
    }
  }

  std::map<char,char> replacer;
  int count = 0;
  for(char c : symbols) {
    replacer[c] = getSymbol(count);
    count++;
  }

  for(char c : einsumString) {
    ret.insert(ret.end(),replacer[c]);
  }

}
void alphaCanoncalize(std::string equation,std::string& ret) {
  std::map<char,char> rename;
  for(char c : equation) {
    if(operators.find(c) >= 0) {
      continue;
    }

    if(!rename.count(c)) {
      rename[c] = getSymbol(rename.size());
    }
  }
  for(char c : equation) {
    ret.insert(ret.end(),rename[c]);
  }
}

void findOutputString(std::string subscripts,std::string& ret) {
  // Get the first occurrence
  size_t pos = subscripts.find(',');
  // Repeat till end is reached
  while( pos != std::string::npos) {
    // Replace this occurrence of Sub String
    subscripts.replace(pos, 1, "");
    // Get the next occurrence from the current position
    pos =subscripts.find(',', pos);
  }

  std::set<char> sortedTmpSubscripts;
  for(char c : subscripts) {
    if(std::count(subscripts.begin(), subscripts.end(), c) == 1)
      sortedTmpSubscripts.insert(c);
  }
  for(char c : sortedTmpSubscripts) {
    ret.insert(ret.end(),c);
  }
}

bool hasOnlyValidEinsumChars(std::string input) {
  bool start = true;
  for (char const &c: input) {
    start = isValidEinSumChar(c);
  }
}

std::tuple<int,int,int> findOutputShape(std::vector<std::string> inputs,std::vector<std::vector<int>> shapes,std::string output) {
  /**
   * """Find the output shape for given inputs, shapes and output string, taking
into account broadcasting.
Examples
--------
>>> oe.parser.find_output_shape(["ab", "bc"], [(2, 3), (3, 4)], "ac")
(2, 4)
# Broadcasting is accounted for
>>> oe.parser.find_output_shape(["a", "a"], [(4, ), (1, )], "a")
(4,)
"""
  return tuple(max(shape[loc] for shape, loc in zip(shapes, [x.find(c) for x in inputs]) if loc >= 0) for c in output)
   */
  std::vector<int> a = {1,2,3,4,5};
  std::vector<int> b = {1,2,3,4,5};
  std::vector<int> c2;
  for(char c : output) {
    for(std::string x : inputs) {
      c2.push_back(x.find(c));
    }
  }

  std::vector<int> shapeRet;
  for(int i = 0; i < inputs.size(); i++) {
    auto shape = shapes[i];
    auto loc = c2[i];
    shapeRet.push_back(shape[loc]);
  }
}

//TODO:
/**
 * def possibly_convert_to_numpy(x: Any) -> Any:
"""Convert things without a 'shape' to ndarrays, but leave everything else.
Examples
--------
>>> oe.parser.possibly_convert_to_numpy(5)
array(5)
>>> oe.parser.possibly_convert_to_numpy([5, 3])
array([5, 3])
>>> oe.parser.possibly_convert_to_numpy(np.array([5, 3]))
array([5, 3])
# Any class with a shape is passed through
>>> class Shape:
...     def __init__(self, shape):
...         self.shape = shape
...
>>> myshape = Shape((5, 5))
>>> oe.parser.possibly_convert_to_numpy(myshape)
<__main__.Shape object at 0x10f850710>
"""

if not hasattr(x, "shape"):
    return np.asanyarray(x)
else:
    return x
 * @param inputShape
 * @param block
 * @return
 */

void convertSubScripts(std::vector<std::string> oldSub,std::map<std::string,std::string> symbolMap,std::string & result) {
  for(std::string s : oldSub) {
    auto insert = symbolMap[s];
    result.insert(result.end(),insert.begin(),insert.end());
  }
}

void convertInterLeavedInput(std::vector<NDArray *> operands,std::tuple<std::string,std::vector<std::string>> result) {
  std::vector<NDArray *> tmpOperands;
  for(NDArray* arr : operands) {
    tmpOperands.push_back(arr);
  }

  std::vector<NDArray *> operandList;
  std::vector<NDArray *> subscriptList;
  for(int p = 0; p < operands.size() / 2; p++) {
    auto removeFirst = tmpOperands[0];
    tmpOperands.erase(tmpOperands.begin());
    auto removeSecond = tmpOperands[0];
    tmpOperands.erase(tmpOperands.begin());
    operandList.push_back(removeFirst);
    subscriptList.push_back(removeSecond);
  }

  NDArray *outputList = nullptr;
  if(!tmpOperands.empty()) {
    outputList = tmpOperands[tmpOperands.size() - 1];
  }

  auto symbolSet = subscriptList;
  std::map<char,NDArray *> symbolMap;
  for(int i = 0; i < symbolSet.size(); i++) {
    symbolMap[getSymbol(i)] = symbolSet[i];
  }

}

/**
 * def parse_einsum_input(operands: Any) -> Tuple[str, str, List[ArrayType]]:
"""
A reproduction of einsum c side einsum parsing in python.
Returns
-------
input_strings : str
    Parsed input strings
output_string : str
    Parsed output string
operands : list of array_like
    The operands to use in the numpy contraction
Examples
--------
The operand list is simplified to reduce printing:
>>> a = np.random.rand(4, 4)
>>> b = np.random.rand(4, 4, 4)
>>> parse_einsum_input(('...a,...a->...', a, b))
('za,xza', 'xz', [a, b])
>>> parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
('za,xza', 'xz', [a, b])
"""

if len(operands) == 0:
    raise ValueError("No input operands")

if isinstance(operands[0], str):
    subscripts = operands[0].replace(" ", "")
    operands = [possibly_convert_to_numpy(x) for x in operands[1:]]

else:
    subscripts, operands = convert_interleaved_input(operands)

# Check for proper "->"
if ("-" in subscripts) or (">" in subscripts):
    invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
    if invalid or (subscripts.count("->") != 1):
        raise ValueError("Subscripts can only contain one '->'.")

# Parse ellipses
if "." in subscripts:
    used = subscripts.replace(".", "").replace(",", "").replace("->", "")
    ellipse_inds = "".join(gen_unused_symbols(used, max(len(x.shape) for x in operands)))
    longest = 0

# Do we have an output to account for?
    if "->" in subscripts:
        input_tmp, output_sub = subscripts.split("->")
        split_subscripts = input_tmp.split(",")
        out_sub = True
    else:
        split_subscripts = subscripts.split(",")
        out_sub = False

    for num, sub in enumerate(split_subscripts):
        if "." in sub:
            if (sub.count(".") != 3) or (sub.count("...") != 1):
                raise ValueError("Invalid Ellipses.")

# Take into account numerical values
            if operands[num].shape == ():
                ellipse_count = 0
            else:
                ellipse_count = max(len(operands[num].shape), 1) - (len(sub) - 3)

            if ellipse_count > longest:
                longest = ellipse_count

            if ellipse_count < 0:
                raise ValueError("Ellipses lengths do not match.")
            elif ellipse_count == 0:
                split_subscripts[num] = sub.replace("...", "")
            else:
                split_subscripts[num] = sub.replace("...", ellipse_inds[-ellipse_count:])

    subscripts = ",".join(split_subscripts)

# Figure out output ellipses
    if longest == 0:
        out_ellipse = ""
    else:
        out_ellipse = ellipse_inds[-longest:]

    if out_sub:
        subscripts += "->" + output_sub.replace("...", out_ellipse)
    else:
# Special care for outputless ellipses
        output_subscript = find_output_str(subscripts)
        normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

        subscripts += "->" + out_ellipse + normal_inds

# Build output string if does not exist
if "->" in subscripts:
    input_subscripts, output_subscript = subscripts.split("->")
else:
    input_subscripts, output_subscript = subscripts, find_output_str(subscripts)

# Make sure output subscripts are in the input
for char in output_subscript:
    if char not in input_subscripts:
        raise ValueError("Output character '{}' did not appear in the input".format(char))

# Make sure number operands is equivalent to the number of terms
if len(input_subscripts.split(",")) != len(operands):
    raise ValueError("Number of einsum subscripts must be equal to the " "number of operands.")

return input_subscripts, output_subscript, operands
 * @param inputShape
 * @param block
 * @return
 */

DECLARE_SHAPE_FN(einsum) {
  std::vector<int> dims;

  if (block.width() == 1) {
    dims = *block.getIArguments();
  } else {
    auto y = INPUT_VARIABLE(1);
    dims = y->template asVectorT<int>();
  }

  auto keepDims = block.numB() ? B_ARG(0) : false;
  auto dtype = block.numD() ? D_ARG(0) : DataType::INT64;

  // we're resolving negative axis here
  helpers::adjustAxis(shape::rank(inputShape->at(0)), dims);

  auto in = inputShape->at(0);
  for (auto d : dims) {
    // we have special case here
    if (d == sd::DataTypeUtils::max<int>()) continue;

    REQUIRE_TRUE(d < shape::rank(in), 0, "ArgAmax: axis can't be above rank")
    REQUIRE_TRUE(in[d + 1] != 0, 0, "ArgAmax: you can't reduce along axis with 0 in shape");
  }

  // special case - output is scalar
  if (dims.empty() || (dims.size() == 1 && dims.at(0) == sd::DataTypeUtils::max<int>())) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(dtype));
  }

  return SHAPELIST(
      ShapeUtils::evalReduceShapeInfo('c', dims, inputShape->at(0), dtype, keepDims, false, block.getWorkspace()));
}
}  // namespace ops
}  // namespace sd

#endif
