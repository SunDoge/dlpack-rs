use std::os::raw::c_void;
// use std::ptr;

pub const DLPACK_VERSION: usize = 020;

/// The device type in DLContext.
#[derive(Debug, Copy, Clone)]
pub enum DLDeviceType {
    /// CPU device
    DLCPU = 1,

    /// CUDA GPU device
    DLGPU = 2,

    /// Pinned CUDA GPU device by cudaMallocHost
    /// DLCPUPinned = DLCPU | DLGPU
    DLCPUPinned = 3,

    /// OpenCL devices.
    DLOpenCL = 4,

    /// Vulkan buffer for next generation graphics.
    DLVulkan = 7,

    /// Metal for Apple GPU.
    DLMetal = 8,

    /// Verilog simulator buffer
    DLVPI = 9,

    /// ROCm GPUs for AMD GPUs
    DLROCM = 10,

    /// Reserved extension device type,
    /// used for quickly test extension device
    /// The semantics can differ depending on the implementation.
    DLExtDev = 12,
}

/// A Device context for Tensor and operator.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLContext {
    /// The device type used in the device.
    device_type: DLDeviceType,

    /// The device index
    device_id: i32,
}

/// The type code options DLDataType.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum DLDataTypeCode {
    DLInt = 0,
    DLUInt = 1,
    DLFloat = 2,
    DLBfloat = 4,
}

/// The data type the tensor can hold.
///
/// Examples
/// - float: type_code = 2, bits = 32, lanes=1
/// - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
/// - int8: type_code = 0, bits = 8, lanes=1
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLDataType {
    /// Type code of base types.
    /// We keep it uint8_t instead of DLDataTypeCode for minimal memory
    /// footprint, but the value should be one of DLDataTypeCode enum values.
    pub code: u8,

    /// Number of bits, common choices are 8, 16, 32.
    pub bits: u8,

    /// Number of lanes in the type, used for vector types.
    pub lanes: u16,
}

/// Plain C Tensor object, does not manage memory.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLTensor {
    /// The opaque data pointer points to the allocated data. This will be
    /// CUDA device pointer or cl_mem handle in OpenCL. This pointer is always
    /// aligned to 256 bytes as in CUDA.
    ///
    /// For given DLTensor, the size of memory required to store the contents of
    /// data is calculated as follows:
    /// ```c
    /// static inline size_t GetDataSize(const DLTensor* t) {
    ///   size_t size = 1;
    ///   for (tvm_index_t i = 0; i < t->ndim; ++i) {
    ///     size *= t->shape[i];
    ///   }
    ///   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
    ///   return size;
    /// }
    pub data: *mut c_void,

    /// The device context of the tensor
    pub ctx: DLContext,

    /// Number of dimensions
    pub ndim: i32,

    /// The data type of the pointer
    pub dtype: DLDataType,

    /// The shape of the tensor
    pub shape: *mut i64,

    /// strides of the tensor (in number of elements, not bytes)
    /// can be NULL, indicating tensor is compact and row-majored.
    pub strides: *mut i64,

    /// The offset in bytes to the beginning pointer to data
    pub byte_offset: u64,
}

// impl Default for DLTensor {
//     fn default() -> Self {
//         DLTensor {
//             data: ptr::null_mut(),
//             ctx: DLContext {
//                 device_type: DLDeviceType::DLCPU,
//                 device_id: 0,
//             },
//             ndim: 0,
//             dtype: DLDataType {
//                 code: DLDataTypeCode::DLFloat as u8,
//                 bits: 32,
//                 lanes: 1,
//             },
//             shape: ptr::null_mut(),
//             strides: ptr::null_mut(),
//             byte_offset: 0,
//         }
//     }
// }

/// C Tensor object, manage memory of DLTensor. This data structure is
/// intended to facilitate the borrowing of DLTensor by another framework. It is
/// not meant to transfer the tensor. When the borrowing framework doesn't need
/// the tensor, it should call the deleter to notify the host that the resource
/// is no longer needed.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DLManagedTensor {
    /// DLTensor which is being memory managed
    pub dl_tensor: DLTensor,

    /// the context of the original host framework of DLManagedTensor in
    /// which DLManagedTensor is used in the framework. It can also be NULL.
    pub manager_ctx: *mut c_void,

    /// Destructor signature void (*)(void*) - this should be called
    /// to destruct manager_ctx which holds the DLManagedTensor. It can be NULL
    /// if there is no way for the caller to provide a reasonable destructor.
    /// The destructors deletes the argument self as well.
    pub deleter: Option<unsafe extern "C" fn(self_: *mut DLManagedTensor)>,
}
