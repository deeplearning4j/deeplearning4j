Onnx op definition loading
---------------------------------

Setup
-------
Use anaconda and install onnx:
```
conda install onnx
```

Generate a file
---------------------
```
python onnx_def_gen.py
```

This will generate a file with all op definitions
loadable as NodeProto in onnx serialized as a text file
split by --\n.
