#pragma once

struct HashTable
{
    uint32_t key;
    uint32_t value;
};

HashTable* CreateTable();

void insert(HashTable* hash_table, const )