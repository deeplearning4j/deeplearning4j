
﻿
# Libnd4j NdArray padded buffers, strides for Arm_Compute Library wrapper

## Status
PROPOSED 

Proposed by: Abdelrauf (23/09/2020)

Discussed with: 

## Context
During the integration process of our library with arm_compute, I faced that our NdArray strides are not flexible. (i.e it cant be set properly without **special and manual handling**). 
Let's say our Nd Array shapes are `[3,4,2]` and the last index is moving faster (i.e C order). Then our strides will be  `[ 8, 2, 1 ]`.
As far as I know, our last index stride can be different (called as ews), but overall strides should follow the cyclic strict rule of dependency.:

    strides[index-1] = strides[index] * shapes[index];
On arm_compute besides strides there is also Padding `{top, right, bottom, left}` that can be used to increase strides and change offsets adn as well as total size. its mostly done for performance reasons. As from above we can see that **its just hosting NdArray shape in the buffer of the bigger NdArray shape**. In arm_compute those paddings applied to last 2 dimensions (on NCHW it will be H and W}.  We can define it like this:

    newH = pad.top + H + pad.bottom;
    newW = pad.left + W + pad.right;

so strides will be calculated for the shape `{N,C, newH, newW}` and offset of the first element will be:

     offset = pad.left * strideOfNewW + pad.top * strideOfNewH


## Proposal
Introduce helper functions checking  below case :

    strides[index-1] >= strides[index] * shapes[index];

Add **generic method for the padded buffer** ( we can simulate arm_compute 2d padding and more)

    int paddings[rank] = {...}; // total padding
    int paddingOffsets[rank] = {...}; //offset indices of the first element

This could be used to padd ndArray shapes and calculate strides based on it while keeping original shape, paddOffsets could be used to determine the beginning of the first element. Though this interface ismore generic its drawback is that on armcompute its possible to padd 1d into 2D while keeping rank but on this one we should supply 2d with one of its dimensions being 1.
   

## Consequences 

 1. All tests that were not tested **against subArray** could break. So they will require a fix
 2. Writing additional test cases 

### Advantages
- alignment possibility for CPUs where alignment is required for speed and vectorization.
- easier integration with libraries. in the case of arm_compute, the last two dimensions are sometimes padded.

  
### Disadvantages
- its advantage is not so big for modern CPUs where unaligned vector loads possible
- exposing it for users is not desirable: (excessive usage creates unnecessary memory spaces and performance problems)
- could result in unnecessary complications for some function implementations
- possibility of requiring additional tests and fixes 


### Technical details about the addition of this functionality into  NdArray
A little investigation showed that the current NdArray actually has constructors to specify strides. 
Here is the constructor that could be used
[ShapeDescriptor.h](https://github.com/KonduitAI/deeplearning4j/blob/qwr_armcompute/libnd4j/include/array/ShapeDescriptor.h)
Here are additions into ShapeDescriptor:
- validate()   //it willbe used for validation of strides and et cetera. This way we can  create NdArray by just using ShapeDescriptor alone. And it will be more flexible with correctness
- allocLength() //returns minimal buffer size for the given strides and shapes. (this was missing on libnd4j side)
- paddedBufferDescriptor(..) //helper method for returning ShapeDescriptor for padded buffer. 



####  [NdArrayFactory](https://github.com/KonduitAI/deeplearning4j/blob/qwr_armcompute/libnd4j/include/array/impl/NDArrayFactory.cpp#L39-L80)
The method that is using ShapeDescriptor validation, and ShapeDescriptor paddedBuffer .

Furthermore to indicate that shape of the NdArray is using paddedBuffer we will flag with `ARRAY_HAS_PADDED_BUFFER` . so it will be possible to know if NdArray  is padded. 

Furthermore, it is still possible to recover Paddings from the allocation size of the padded NdArray. But its not an easy task to get PaddingOffsets from offset and recovered full shape. Thats why it requires storing them. Fortunately, for arm_compute tensors **manual padding** we just need to know **total size and the offset** of the first element. So we dont need to change internals that much

As our padded Buffer follows the strict ews() rule instead of the loose one. Paddings will be obtained from this rule:

    strides[index-1] == strides[index] * shapes[index];

pseudo code for C order:

    for (int j = rank - 1; j >= 0; j--) {
        shapesAfterPadding[j] = strides[j - 1] / strides[j]
    }
    shapesAfterPadding[0] = buffer.AllocSize / strides[0]
    //Paddings for index in 0..rank-1
    paddings[index] = shapesAfterPadding[index] - shape[index] 





### Technical notes on arm_compute library

The main drive for the above proposal to avoid unnecessary performance and memory allocation. And also we should keep on mind :
- in each newer version of arm_compute there are new implementations in which the padding requirements were removed. 

This **can diminish the necessity for the proposed changes** if such versions of the desired functions are implemented. 

##### Notes on  arm_compute tensors
Arm_compute tensors are mostly 3d 4d with max 6d dimensions. 
So lets show  C order NdArray({2,2,5,5},) 

    shapeInfo shapeInfo: [4,  2,2,5,5,  50,25,5,1,  8192,1,99]

of float type and its arm_compute tensor equivalent :
- first of all, we map NdArray dataTypes into arm_compute [armcomputeUtils.cpp#L35-L75](https://github.com/KonduitAI/deeplearning4j/blob/qwr_armcompute/libnd4j/include/ops/declarable/platform/armcompute/armcomputeUtils.cpp#L35-L75)
- it will be with the reversed shape. **`NdArray{n,z,y,x} -> TensorShape{x,y,z,n}`** 
- 

    total length in bytes: 400
    shapes: 5,5,2,2,1,1,
    strides in bytes: 4,20,100,200,0,0,  
    strides as elements: (1,5,25,50)

Paddings in arm_compute Tensors. `Padding{left,right, top, bottom}`
As both OpenCL and NEON use vector loads and stores instructions to access the data in buffers, so in order to avoid having special cases to handle for the borders all the images and tensors used in this library must be padded
There are different ways padding can be calculated:

-   Accurate padding. 
 in this case it is importan to configure and then after that to  allocate  
-  auto padding. 
  It guarantees that the allocation will have enough padding to run any of the provided functions
- no padding
- manual padding

#### how padding affects strides offset and total size
in arm_compute Tensor:
it's 2d  {Width Height} can be padded and thats why it affects strides.
Lets show it with the picture:

       \            top          /
        \ _____________________ /
    left |          ^           | right
         |          Width       |
         | <-Height             |
         |                      |
         |                      |
          ----------------------
        /       bottom           \
       /                          \
         
Here is the stride calculation pseudo code for Tensor {x,y,z}  

    stride_x = element_size(); //float will be 4
    stride_y = (padding.left + _tensor_shape[0] + padding.right) * stride_x;
    stride_z = (padding.top + _tensor_shape[1] + padding.bottom) * stride_y;
    
    required_offset_first_element = padding.left * stride_x + padding.top * stride_y;

  
For example: if arm_tensor had `padding: left 0, right 1, top 0, bottom 1` :

    total: 576
    shapes: 5,5,2,2,1,1,
    strides in bytes: 4,24,144,288,0,0,



### Notes on the current wrapper implementation

This is a simple wrapper for arm functions with input and output tensors:
[armcomputeUtils.h#L95-L165](https://github.com/KonduitAI/deeplearning4j/blob/qwr_armcompute/libnd4j/include/ops/declarable/platform/armcompute/armcomputeUtils.h#L85-L133)

From above we could see :
- we had to flag padded NdArrays so that we can use manual padding version of arm_compute Tensors
- when padding information is changed during configure process we **have to copy** our NdArray buffer into **new allocated** arm_tensor buffer.  and the same with the output.
-  for cases without padding , arm_tensor could use our buffer if its ews()==1.
- its desired to call configure and run  separately to avoid multiple configure calls ( this is not discussed here, for now)


## arm_compute wrapper proposal   


So from above we can conclude that we have two options:

- creating our NdArray with auto_padding  strides and modifying the current wrapper. Still configure will be called foreach run. But with auto padding it is using more memory for small ndarrays
- to be able to use accurate padding properly we should call configure before NdArray memory allocation so that we can import it. For that I should investigate graph, DeclarableOps and NdArrays  usage lifecycle.

Here is auto padding:

    // Some kernels compute 32 elements at the time, worst case scenario they
    // will read 32 values after the last element
    extra_pad_x = _tensor_shape.num_dimensions() < 1 ? 0 : 32;
    pad_x       = _tensor_shape.num_dimensions() < 1 ? 0 : 4;
    pad_y       = _tensor_shape.num_dimensions() < 2 ? 0 : 4;
    
    PaddingSize(pad_y, pad_x + extra_pad_x, pad_y, pad_x);

## Discussion






