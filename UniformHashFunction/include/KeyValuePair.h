#pragma once

#include <iostream>
#include <stdalign.h>
#include "alignment.h"

template <typename Key, typename Value, bool Padding = padding_size<Key, Value>()!=0>
struct alignas (pair_align<Key,Value>()) Pair{
    Key key;
    Value value;

    Pair() = default;

    constexpr Pair(const Key& k, const Value& v): key(k), value(v) {}

    constexpr bool operator==(const Pair& rhs) const{
        return (key == rhs.key) && (value == rhs.value);
    }

    constexpr bool operator!=(const Pair& rhs) const{
        return (key!=rhs.key) && (value != rhs.value);
    }
};

//When padding is needed
template <typename Key, typename Value>
struct alignas (pair_align<Key, Value>()) Pair<Key, Value, true>{
    Key key;
    Value value;

    //if padding is needed include the bits on struct
    private:
        char padding_bits[padding_size<Key, Value>()] = {0};
    public:
        Pair() = default;

        constexpr Pair(const Key& k, const Value& v): key(k), value(v) {}

        constexpr bool operator==(const Pair& rhs) const{
            return (key == rhs.key) && (value == rhs.value);
        }

        constexpr bool operator!=(const Pair& rhs) const{
            return (key!=rhs.key) && (value != rhs.value);
        }
};
