use super::{DLContext, DLDataType, DLDataTypeCode, DLDeviceType, DLTensor};
use std::fmt;
use std::ptr;
use std::slice;

impl Default for DLDeviceType {
    fn default() -> Self {
        DLDeviceType::DLCPU
    }
}

impl Default for DLDataTypeCode {
    fn default() -> Self {
        DLDataTypeCode::DLFloat
    }
}

impl From<(u8, u8, u16)> for DLDataType {
    fn from(dtype: (u8, u8, u16)) -> Self {
        let (code, bits, lanes) = dtype;
        DLDataType { code, bits, lanes }
    }
}

impl From<(DLDataTypeCode, u8, u16)> for DLDataType {
    fn from(dtype: (DLDataTypeCode, u8, u16)) -> Self {
        let (code, bits, lanes) = dtype;
        DLDataType {
            code: code as u8,
            bits,
            lanes,
        }
    }
}

impl DLDataType {
    pub fn new(code: DLDataTypeCode, bits: u8, lanes: u16) -> DLDataType {
        DLDataType {
            code: code as u8,
            bits,
            lanes,
        }
    }
}

impl Default for DLDataType {
    fn default() -> Self {
        DLDataType {
            code: DLDataTypeCode::DLFloat as u8,
            bits: 32,
            lanes: 1,
        }
    }
}

impl Default for DLTensor {
    fn default() -> Self {
        DLTensor {
            data: ptr::null_mut(),
            ctx: DLContext::default(),
            ndim: 0,
            dtype: DLDataType::default(),
            shape: ptr::null_mut(),
            strides: ptr::null_mut(),
            byte_offset: 0,
        }
    }
}

impl fmt::Debug for DLTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DLTensor")
            .field("data", &self.data)
            .field("ctx", &self.ctx)
            .field("ndim", &self.ndim)
            .field("dtype", &self.dtype)
            .field("shape", &unsafe {
                slice::from_raw_parts_mut(self.shape , self.ndim as usize)
            })
            .field("strides", &unsafe {
                slice::from_raw_parts_mut(self.strides, self.ndim as usize)
            })
            .field("byte_offset", &self.byte_offset)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_device() {
        assert_eq!(DLDeviceType::default(), DLDeviceType::DLCPU);
    }

    #[test]
    fn default_context() {
        let ctx = DLContext::default();
        assert_eq!(ctx.device_type, DLDeviceType::default());
        assert_eq!(ctx.device_id, 0);
    }
}
