#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

// 1-2
bool Tensor::isContiguous() const {
    //TO_BE_IMPLEMENTED();
    if (_meta.shape.empty() || _meta.shape.size() == 1) {
        return true;
    }

    if (_meta.strides.back() != 1) {
        return false;
    }

    size_t expected_stride = 1;
    for (int i = _meta.shape.size() - 2; i >= 0; --i) {
        expected_stride *= _meta.shape[i + 1];
        if (_meta.strides[i] != static_cast<ptrdiff_t>(expected_stride)) {
            return false;
        }
    }

    return true;
}

// 1-4
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    //TO_BE_IMPLEMENTED();
    if (order.size() != this->ndim()) {
        throw std::invalid_argument("Invalid permute order: size does not match tensor dimensions.");
    }
    std::vector<bool> seen(this->ndim(), false);
    for (size_t dim : order) {
        if (dim >= this->ndim() || seen[dim]) {
            throw std::invalid_argument("Invalid permute order: contains duplicates or out of bounds.");
        }
        seen[dim] = true;
    }

    TensorMeta new_meta;
    new_meta.dtype = this->_meta.dtype;
    new_meta.shape.resize(this->ndim());
    new_meta.strides.resize(this->ndim());

    for (size_t i = 0; i < order.size(); ++i) {
        new_meta.shape[i] = this->_meta.shape[order[i]];
        new_meta.strides[i] = this->_meta.strides[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
    //return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

// 1-3
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    //TO_BE_IMPLEMENTED();

    size_t new_elements = 1;

    for (const auto& dim_size : shape) {
        new_elements *= dim_size;
    }
    if (new_elements != this->numel()) {
        throw std::runtime_error("View shape does not match the number of elements.");
    }

    if (!this->isContiguous()) {
        throw std::runtime_error("View is not supported on non-contiguous tensor. "
                                 "Use reshape() or contiguous().view() instead.");
    }

    TensorMeta new_meta;
    new_meta.dtype = this->_meta.dtype;
    new_meta.shape = shape;

    new_meta.strides.resize(shape.size());
    if (!shape.empty()) {
        new_meta.strides.back() = 1;
        for (int i = shape.size() - 2; i >= 0; --i) {
            new_meta.strides[i] = new_meta.strides[i + 1] * shape[i + 1];
        }
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
    //return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

// 1-5
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    //TO_BE_IMPLEMENTED();
    if (dim >= this->ndim()) {
        throw std::out_of_range("Dimension out of range for slice.");
    }
    if (start >= end || end > this->_meta.shape[dim]) {
        throw std::out_of_range("Slice indices out of range.");
    }

    TensorMeta new_meta = this->_meta;

    size_t new_offset = this->_offset + start * this->_meta.strides[dim] * this->elementSize();

    new_meta.shape[dim] = end - start;


    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
    //return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

// 1-1
// 把一块已有的数据（通常在 CPU 主机内存里）拷贝到当前 Tensor 的设备内存里。
void Tensor::load(const void *src_) {
    //TO_BE_IMPLEMENTED();
    auto& ctx = core::context();

    ctx.setDevice(this->deviceType(), this->deviceId());

    auto& runtime = ctx.runtime();
    auto api = runtime.api();

    size_t size = this->numel() * this->elementSize();

    api->memcpy_sync(   this->data(), 
                        src_, 
                        size, 
                        LLAISYS_MEMCPY_H2D);

    std::cout << "Data loaded to tensor, size: " << size << " bytes." << std::endl; 
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
