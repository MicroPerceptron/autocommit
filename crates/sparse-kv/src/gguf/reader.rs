use std::io::{BufReader, Read, Seek};
use std::path::Path;

use memmap2::Mmap;

use crate::error::InferenceError;
use crate::quant::types::QuantType;
use crate::quant::view::QuantSlice;

use super::metadata::{GgufMetadata, GgufValue};
use super::weights::{TensorInfo, WeightMap};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

pub struct GgufReader {
    pub metadata: GgufMetadata,
    pub weight_map: WeightMap,
    data_mmap: Mmap,
    data_offset: u64,
    // Keep the file handle alive for the mmap's lifetime (required on Windows).
    _file: std::fs::File,
}

impl GgufReader {
    pub fn open(path: &Path) -> Result<Self, InferenceError> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut r = BufReader::new(&file);

        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(InferenceError::Load(format!(
                "not a GGUF file (magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
            )));
        }

        let version = read_u32(&mut r)?;
        if version < 2 || version > 3 {
            return Err(InferenceError::Load(format!(
                "unsupported GGUF version: {version} (expected 2 or 3)"
            )));
        }

        let tensor_count = read_u64(&mut r)?;
        let metadata_kv_count = read_u64(&mut r)?;

        let mut metadata = GgufMetadata::default();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut r)?;
            let value = read_value(&mut r)?;
            metadata.entries.insert(key, value);
        }

        let mut weight_map = WeightMap::default();
        let mut max_tensor_end: u64 = 0;

        for _ in 0..tensor_count {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let type_id = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;

            let quant_type = QuantType::from_ggml_type(type_id).ok_or_else(|| {
                InferenceError::Load(format!(
                    "tensor '{name}' has unsupported ggml type: {type_id}"
                ))
            })?;

            let n_elements = dims.iter().product::<u64>().max(1) as usize;
            let size = quant_type.row_size(dims.first().copied().unwrap_or(1) as usize)
                * (n_elements / dims.first().copied().unwrap_or(1).max(1) as usize);

            let tensor_end = offset + size as u64;
            if tensor_end > max_tensor_end {
                max_tensor_end = tensor_end;
            }

            weight_map.entries.insert(
                name,
                TensorInfo {
                    quant_type,
                    dims,
                    offset,
                    size,
                },
            );
        }

        // Data section starts at the current position, aligned to 32 bytes
        let header_end = r.stream_position()?;
        let alignment = metadata
            .get_u32("general.alignment")
            .unwrap_or(32) as u64;
        let data_offset = (header_end + alignment - 1) / alignment * alignment;

        // Only validate data bounds when tensors are present.
        // Vocab-only GGUF files have no tensors, so the aligned data_offset
        // may legally exceed the file length.
        if max_tensor_end > 0 && data_offset + max_tensor_end > file_len {
            return Err(InferenceError::Load(format!(
                "GGUF data section extends beyond file (data_offset={data_offset}, \
                 max_tensor_end={max_tensor_end}, file_len={file_len})"
            )));
        }

        // SAFETY: memory-mapping the data section of a GGUF file that we
        // verified fits within the file bounds.
        let data_mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                InferenceError::Load(format!("failed to mmap GGUF data: {e}"))
            })?
        };

        Ok(Self {
            metadata,
            weight_map,
            data_mmap,
            data_offset,
            _file: file,
        })
    }

    /// Get a zero-copy view of a tensor's data.
    pub fn tensor_data(&self, name: &str) -> Result<QuantSlice<'_>, InferenceError> {
        let info = self.weight_map.get(name).ok_or_else(|| {
            InferenceError::Load(format!("tensor '{name}' not found in GGUF"))
        })?;

        let start = self.data_offset as usize + info.offset as usize;
        let end = start + info.size;

        if end > self.data_mmap.len() {
            return Err(InferenceError::Load(format!(
                "tensor '{name}' data out of bounds (start={start}, size={}, mmap_len={})",
                info.size,
                self.data_mmap.len()
            )));
        }

        let data = &self.data_mmap[start..end];
        Ok(QuantSlice::new(info.quant_type, info.shape_2d(), data))
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }
}

fn read_u8(r: &mut impl Read) -> Result<u8, InferenceError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> Result<i8, InferenceError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> Result<u16, InferenceError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> Result<i16, InferenceError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32, InferenceError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32, InferenceError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64, InferenceError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64, InferenceError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> Result<f32, InferenceError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, InferenceError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String, InferenceError> {
    let len = read_u64(r)? as usize;
    if len > 1 << 24 {
        return Err(InferenceError::Load(format!(
            "GGUF string too long: {len} bytes"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| InferenceError::Load(format!("invalid UTF-8 in GGUF string: {e}")))
}

fn read_value(r: &mut impl Read) -> Result<GgufValue, InferenceError> {
    let type_id = read_u32(r)?;
    read_value_of_type(r, type_id)
}

fn read_value_of_type(r: &mut impl Read, type_id: u32) -> Result<GgufValue, InferenceError> {
    Ok(match type_id {
        GGUF_TYPE_U8 => GgufValue::U8(read_u8(r)?),
        GGUF_TYPE_I8 => GgufValue::I8(read_i8(r)?),
        GGUF_TYPE_U16 => GgufValue::U16(read_u16(r)?),
        GGUF_TYPE_I16 => GgufValue::I16(read_i16(r)?),
        GGUF_TYPE_U32 => GgufValue::U32(read_u32(r)?),
        GGUF_TYPE_I32 => GgufValue::I32(read_i32(r)?),
        GGUF_TYPE_U64 => GgufValue::U64(read_u64(r)?),
        GGUF_TYPE_I64 => GgufValue::I64(read_i64(r)?),
        GGUF_TYPE_F32 => GgufValue::F32(read_f32(r)?),
        GGUF_TYPE_F64 => GgufValue::F64(read_f64(r)?),
        GGUF_TYPE_BOOL => GgufValue::Bool(read_u8(r)? != 0),
        GGUF_TYPE_STRING => GgufValue::String(read_string(r)?),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)? as usize;
            if len > 1 << 24 {
                return Err(InferenceError::Load(format!(
                    "GGUF array too long: {len} elements"
                )));
            }
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value_of_type(r, elem_type)?);
            }
            GgufValue::Array(items)
        }
        _ => {
            return Err(InferenceError::Load(format!(
                "unknown GGUF value type: {type_id}"
            )));
        }
    })
}
