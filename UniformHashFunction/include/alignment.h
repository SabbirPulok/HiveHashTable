#pragma once

#include <iostream>

template<typename T>
constexpr std::size_t next_align();

template<typename T1, typename T2>
constexpr std::size_t pair_size();


template<typename T1, typename T2>
constexpr std::size_t pair_align();

template<typename T1, typename T2>
constexpr std::size_t padding_size();