#pragma once

#include <cassert>
#include <cstdlib>
#include "Collection.h"

namespace TurboCV
{
namespace System
{
    inline ArrayList<size_t> RandomPermutate(size_t cardNum, size_t pickUpNum)
    {
        assert(cardNum >= pickUpNum);
        ArrayList<size_t> result;

        size_t* cards = new size_t[cardNum];
        for (size_t i = 0; i < cardNum; i++)
            cards[i] = i;

        for (size_t i = 0; i < pickUpNum; i++)
        {
            size_t index = (size_t)((double)rand() * (cardNum - i - 1) / RAND_MAX + i);
            assert(index < cardNum);
            std::swap(cards[i], cards[index]);
        }

        for (size_t i = 0; i < pickUpNum; i++)
            result.Add(cards[i]);

        delete[] cards;
        return result;
    }

    template<typename T>
    inline ArrayList<T> PickUp(const ArrayList<T>& vec, const ArrayList<size_t>& pickUpIndexes)
    {
        ArrayList<size_t> indexes = pickUpIndexes;
        std::sort(indexes.Begin(), indexes.End());

        ArrayList<T> pickUps;
        size_t cardNum = vec.Count(), counter = 0;

        for (size_t i = 0; i < cardNum; i++)
        {
            if (counter < indexes.Count() && indexes[counter] == i)
            {
                counter++;
                pickUps.Add(vec[i]);
            }
        }

        return pickUps;
    }

    template<typename T>
    inline Tuple<ArrayList<T>, ArrayList<T>, ArrayList<size_t>> Divide(
        const ArrayList<T>& vec, const ArrayList<size_t>& pickUpIndexes)
    {
        ArrayList<size_t> indexes = pickUpIndexes;
        std::sort(indexes.begin(), indexes.end());

        ArrayList<T> pickUps, others;
        size_t cardNum = vec.Count(), counter = 0;

        for (size_t i = 0; i < cardNum; i++)
        {
            if (counter < indexes.Count() && indexes[counter] == i)
            {
                counter++;
                pickUps.Add(vec[i]);
            }
            else
                others.Add(vec[i]);
        }

        return CreateTuple(pickUps, others, indexes);
    }

    template<typename T>
    inline ArrayList<T> RandomPickUp(const ArrayList<T>& vec, size_t pickUpNum)
    {
        size_t cardNum = vec.Count();
        assert(cardNum >= pickUpNum);

        return PickUp(vec, RandomPermutate(cardNum, pickUpNum));
    }

    template<typename T>
    inline ArrayList<Tuple<ArrayList<T>, ArrayList<T>, ArrayList<size_t>>> RandomSplit(
        const ArrayList<T>& vec, size_t fold)
    {
        size_t cardNum = vec.Count();
        assert(cardNum >= fold);

        ArrayList<size_t> permutation = RandomPermutate(cardNum, cardNum);

        ArrayList<Tuple<ArrayList<T>, ArrayList<T>, ArrayList<size_t>>> result;
        for (size_t i = 0; i < fold; i++)
        {
            ArrayList<size_t> subsetIndexes;
            size_t begin = cardNum / fold * i,
                end = (i != fold - 1) ? cardNum / fold * (i + 1) : cardNum;

            for (size_t j = begin; j < end; j++)
                subsetIndexes.Add(permutation[j]);
                
            result.Add(Divide(vec, subsetIndexes));
        }

        return result;
    }
}
}