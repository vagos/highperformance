#pragma once

#include <bits/stdc++.h>
#include <ios>

#define IDX(X, Y) ((X) + (Y * W))

template <typename T>
void printArray(std::ostream& stream, T* a, std::size_t W, std::size_t H)
{
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            stream << a[IDX(x, y)] << ' ';
        }

        stream << '\n';
    }
}

