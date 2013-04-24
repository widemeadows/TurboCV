#pragma once

#include <cassert>
#include <cstdlib>
#include "Type.h"

namespace TurboCV
{
namespace System
{
    template<typename RandomAccessIterator, typename T> 
    bool Contains(
        const RandomAccessIterator& begin, 
        const RandomAccessIterator& end, 
        const T& item)
    {
        RandomAccessIterator itr = begin;

        while (itr != end)
        {
            if (*itr == item)
                return true;
            itr++;
        }

        return false;
    }

    inline Vector<size_t> RandomPermutate(
        size_t cardNum, 
        size_t pickUpNum)
    {
        assert(cardNum >= pickUpNum);
        Vector<size_t> result;

        size_t* cards = new size_t[cardNum];
        for (size_t i = 0; i < cardNum; i++)
            cards[i] = i;

        for (size_t i = 0; i < pickUpNum; i++)
        {
            size_t index = (size_t)
                ((double)rand() * (cardNum - i - 1) / RAND_MAX + i);
            assert(index < cardNum);
            swap(cards[i], cards[index]);
        }

        for (size_t i = 0; i < pickUpNum; i++)
            result.push_back(cards[i]);

        delete[] cards;
        return result;
    }

    template<typename T>
    inline Vector<T> PickUp(
        const Vector<T>& vec, 
        const Vector<size_t>& pickUpIndexes)
    {
        Vector<size_t> indexes = pickUpIndexes;
        sort(indexes.begin(), indexes.end());

        Vector<T> pickUps;
        size_t cardNum = vec.size(), counter = 0;

        for (size_t i = 0; i < cardNum; i++)
        {
            if (counter < indexes.size() && indexes[counter] == i)
            {
                counter++;
                pickUps.push_back(vec[i]);
            }
        }

        return pickUps;
    }

    template<typename T>
    inline Tuple<Vector<T>, Vector<T>, Vector<size_t>> Divide(
        const Vector<T>& vec, 
        const Vector<size_t>& pickUpIndexes)
    {
        Vector<size_t> indexes = pickUpIndexes;
        sort(indexes.begin(), indexes.end());

        Vector<T> pickUps, others;
        size_t cardNum = vec.size(), counter = 0;

        for (size_t i = 0; i < cardNum; i++)
        {
            if (counter < indexes.size() && indexes[counter] == i)
            {
                counter++;
                pickUps.push_back(vec[i]);
            }
            else
                others.push_back(vec[i]);
        }

        return CreateTuple(pickUps, others, indexes);
    }

    template<typename T>
    inline Vector<T> RandomPickUp(
        const Vector<T>& vec, 
        size_t pickUpNum)
    {
        size_t cardNum = vec.size();
        assert(cardNum >= pickUpNum);

        return PickUp(vec, RandomPermutate(cardNum, pickUpNum));
    }

    template<typename T>
    inline Vector<Tuple<Vector<T>, Vector<T>, Vector<size_t>>> RandomSplit(
        const Vector<T>& vec, 
        size_t fold)
    {
        size_t cardNum = vec.size();
        assert(cardNum >= fold);

        Vector<size_t> permutation = RandomPermutate(cardNum, cardNum);

        Vector<Tuple<Vector<T>, Vector<T>, Vector<size_t>>> result;
        for (size_t i = 0; i < fold; i++)
        {
            Vector<size_t> subsetIndexes;
            size_t begin = cardNum / fold * i,
                end = (i != fold - 1) ? cardNum / fold * (i + 1) : cardNum;

            for (size_t j = begin; j < end; j++)
                subsetIndexes.push_back(permutation[j]);
                
            result.push_back(Divide(vec, subsetIndexes));
        }

        return result;
    }
}
}