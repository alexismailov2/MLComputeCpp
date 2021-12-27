#pragma once

#import <cstdint>

#include <iostream>

class CppMLCTensor;
class CppMLCTypesPrivate;

class CppMLCTensorData
{
public:
    /*! @abstract   Creates a data object that holds a given number of bytes from a given buffer.
        @note       The returned object will not take ownership of the \p bytes pointer and thus will not free it on deallocation.
        @param      bytes   A buffer containing data for the new object.
        @param      length  The number of bytes to hold from \p bytes. This value must not exceed the length of \p bytes.
        @return     A new \p MLCTensorData object.
     */
    CppMLCTensorData(void* bytes, uint64_t length);

    /*! @abstract   Creates a data object that holds a given number of bytes from a given buffer.
        @note       The returned object will not take ownership of the \p bytes pointer and thus will not free it on deallocation. The underlying bytes in the return object should not be mutated.
        @param      bytes   A buffer containing data for the new object.
        @param      length  The number of bytes to hold from \p bytes. This value must not exceed the length of \p bytes.
        @return     A new \p MLCTensorData object.
     */
    CppMLCTensorData(void const* bytes, uint64_t length);

    /*! @property   bytes
        @abstract   Pointer to memory that contains or will be used for tensor data
     */
    auto getBytes() const -> void const*;

    /*! @property   length
        @abstract   The size in bytes of the tensor data
     */
    auto getLength() const -> uint64_t;

private:
    CppMLCTensorData(void* self);

private:
    void* self;
    friend CppMLCTensor;
    friend CppMLCTypesPrivate;
};

std::ostream& operator<<(std::ostream&, CppMLCTensorData const& tensorData);