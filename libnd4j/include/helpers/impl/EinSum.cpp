//
// Created by agibsonccc on 9/10/22.
//
#include <ops/declarable/EinSum.h>
/**
 * A port of opteinsum to c++, please see:
 * https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/
 */
using namespace sd;
namespace samediff {
namespace einsum {
bool isValidEinSumChar(const char input) {
  return validString.find(input) != std::string::npos
         && operators.find(input) != std::string::npos;
}

char getSymbol(int i) {
  return (char) i;
}

std::vector<char> getUnused(std::string used,int n,std::vector<std::string> ret) {
  int i,count = 0;
  while(count < n) {
    char currChar = getSymbol(i);
    i += 1;
    if(used.find(currChar) >= 0)
      continue;
    ret.push_back("" + currChar);
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

void split(std::string input,std::string stringDelimiter,std::vector<std::string> result) {
  auto end = input.find(stringDelimiter);
  while (end != std::string::npos) {
    auto start = 0U;
    result.push_back(input.substr(start, end - start) );
    start = end + stringDelimiter.length();
    end = input.find(stringDelimiter, start);
  }
}

void join(std::string result,int beginIdx,int endIndx,std::vector<std::string> joinInput) {
  for(int i = beginIdx; i < endIndx; i++) {
    result.append(joinInput[i]);
  }
}

void joinWithDelimiter(std::string delimiter,std::vector<std::string> toJoin,std::string& result) {
  for(int i = 0; i < toJoin.size(); i++) {
    auto element = toJoin[i];
    result.append(element);
    if(i < toJoin.size() - 1) {
      result.append(delimiter);
    }
  }
}

void parseEinsumInput(std::string inputOperands,
                      std::vector<NDArray *> operands,
                      std::string & inputSubscriptsResult,
                      std::string & outputSubscriptResult,
                      std::vector<NDArray *> & operandsOutput) {
  // void convertInterLeavedInput(std::vector<NDArray *> operands,std::tuple<std::string,std::vector<std::string>> result) {
  std::tuple<std::string,std::vector<std::string>> result;
  convertInterLeavedInput(operands,result);
  auto subscripts = get<0>(result);
  auto operands2 = get<1>(result);
  if(subscripts.find('-') || subscripts.find('>')) {
    if(std::count(subscripts.begin(), subscripts.end(), '-') > 1 ||
        std::count(subscripts.begin(),subscripts.end(),'>') > 1) {
      throw std::runtime_error("Invalid expression passed in. Please ensure both - and > only occur once in the expression");
    } else {
      auto count = 0;
      for(int i = 1; i < subscripts.size(); i++) {
        if(subscripts[i] == '>' && subscripts[i - 1] == '-') {
          count++;
        }
      }
      if(count != 1) {
        throw std::runtime_error("Invalid expression: -> must only occur once in an einsum expression.");
      }
    }

    if(subscripts.find('.') >= 0) {
      //replace with empty character
      std::replace(subscripts.begin(),subscripts.end(),'.','\0');
      std::replace(subscripts.begin(),subscripts.end(),',','\0');
      auto idx = subscripts.find("->");
      subscripts.replace(subscripts.begin() + idx,subscripts.end(),"");
      std::replace(subscripts.begin(),subscripts.end(),'.','\0');
      sd::LongType maxLength = 0;
      for(auto arr : operands) {
        maxLength = std::max(maxLength,arr->lengthOf());
      }
      std::vector<std::string> ellipseIndices;
      getUnused(subscripts,maxLength, ellipseIndices);
      sd::LongType  longest = 0;
      std::vector<std::string> splitSubScripts;
      std::string outputSub = "";
      bool outSub = false;
      if(subscripts.find("->")) {
        std::vector<std::string> splitTmp;
        split(subscripts,"->",splitTmp);
        auto inputTmp = splitTmp[0];
        outputSub = splitTmp[1];
        split(inputTmp,",",splitSubScripts);
        outSub = true;
      } else {
        split(subscripts,",",splitSubScripts);
        outSub = false;
      }

      auto ellipseCount = 0;
      for(int i = 0; i < splitSubScripts.size(); i++) {
        auto sub = splitSubScripts[i];
        if(std::count(sub.begin(), sub.end(), '.') != 3) {
          throw std::runtime_error("Invalid Ellipses");
        }

        if(operands[i]->rankOf() < 1) {
          ellipseCount = 0;
        } else {
          ellipseCount = std::max(operands[i]->rankOf(),1) - (sub.length() - 3);
        }

        if(ellipseCount > longest) {
          longest = ellipseCount;
        }

        if(ellipseCount < 0) {
          throw std::runtime_error("Ellipses lengths do not match.");
        } else if(ellipseCount == 0) {
          auto idx = subscripts.find("...");
          splitSubScripts[i].replace(subscripts.begin() + idx,subscripts.end(),"");;
          subscripts.replace(subscripts.begin() + idx,subscripts.end(),"");
        } else {
          auto idx = subscripts.find("...");
          splitSubScripts[i].replace(subscripts.begin() + idx,subscripts.end(),"");
          auto result = "";
          //    split_subscripts[num] = sub.replace("...", ellipse_inds[-ellipse_count:])
          //TODO: check if join needs to be iterated in reverse
          join(result,0,ellipseCount,ellipseIndices);
          subscripts.replace(subscripts.begin() + idx,subscripts.end(),"");
        }
      }

      auto join2 = "";
      auto casted = (std::vector<std::string>) splitSubScripts;
      joinWithDelimiter(std::string(","),casted, (std::string &)result);
      //TODO: left off on subscripts = ",".join(split_subscripts) line 343
      //https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/parser.py
      std::string outEllipse;
      if(longest == 0) {
        outEllipse = "";
      } else {
        outEllipse = "";
        //out_ellipse = ellipse_inds[-longest:]
        join(outEllipse,0,longest,ellipseIndices);
      }

      std::string outputSubsScript = "";
      if(outSub) {
        subscripts.append("->");
        auto idx = subscripts.find("->");
        outputSub.replace(outputSub.begin() + idx,outputSub.end(),"");
        subscripts.append(outputSub);
      } else {
        findOutputString(subscripts,outputSubsScript);
        std::string normalInds = "";
        for(auto outputSubScriptChar : outputSubsScript) {
          if(!outEllipse.find(outputSubScriptChar)) {
            normalInds.append(outputSubScriptChar + "");
          }
        }

        std::sort(normalInds.begin(), normalInds.end());
        subscripts.append("->");
        subscripts.append(outEllipse);
        subscripts.append(normalInds);
        //normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))
        // subscripts += "->" + out_ellipse + normal_inds
      }

      std::string inputSubscripts = "";
      std::string outputSubscript = "";
      //https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/parser.py#L361
      if(subscripts.find("->")) {
        std::vector<std::string> splitTmp;
        split(subscripts,"->",splitTmp);
        inputSubscripts = splitTmp[0];
        outputSubsScript = splitTmp[1];
      } else {
        inputSubscripts = subscripts;
        findOutputString(subscripts,outputSubsScript);
      }

      for(auto in:  outputSubsScript) {
        if(!inputSubscripts.find(in)) {
          throw std::runtime_error("Output character did not appear in the input");
        }
      }
      std::vector<std::string> operandValidation;
      split(inputSubscripts,",",operandValidation);
      if(operandValidation.size() != operands.size()) {
        throw std::runtime_error("Final operands did not match input operands");
      }

      inputSubscriptsResult.append(inputSubscripts);
      outputSubscriptResult.append(outputSubsScript);
      for(NDArray * arr: operands)
        operandsOutput.push_back(arr);

    }
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
}
}
