#pragma once

#include <vector>
#include <cassert>
using namespace std;

namespace System
{
    inline vector<int> RandomPermutate(size_t cardNum, size_t pickUpNum)
    {
        assert(cardNum >= pickUpNum);
        vector<int> result;

        int* cards = new int[cardNum];
        for (size_t i = 0; i < cardNum; i++)
            cards[i] = i;

        for (size_t i = 0; i < pickUpNum; i++)
        {
            int index = (int)((double)rand() * (cardNum - i - 1) / RAND_MAX + i);
            assert(index >= 0 && index < cardNum);
            swap(cards[i], cards[index]);
        }

        for (size_t i = 0; i < pickUpNum; i++)
            result.push_back(cards[i]);

        delete[] cards;
        return result;
    }

    template<typename T>
    inline vector<T> PickUp(const vector<T>& vec, const vector<int>& pickUpIndexes)
    {
        vector<int> indexes = pickUpIndexes;
        sort(indexes.begin(), indexes.end());

        vector<T> pickUps;
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
    inline tuple<vector<T>, vector<T>, vector<int>> Divide(const vector<T>& vec, 
        const vector<int>& pickUpIndexes)
    {
        vector<int> indexes = pickUpIndexes;
        sort(indexes.begin(), indexes.end());

        vector<T> pickUps, others;
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

        return make_tuple(pickUps, others, indexes);
    }

    template<typename T>
    inline vector<T> RandomPickUp(const vector<T>& vec, size_t pickUpNum)
    {
        size_t cardNum = vec.size();
        assert(cardNum >= pickUpNum);

        return PickUp(vec, RandomPermutate(cardNum, pickUpNum));
    }

    template<typename T>
    inline vector<tuple<vector<T>, vector<T>, vector<int>>> RandomSplit(
        const vector<T>& vec, size_t fold)
    {
        size_t cardNum = vec.size();
        assert(cardNum >= fold);

        vector<int> permutation = RandomPermutate(cardNum, cardNum);

        vector<tuple<vector<T>, vector<T>, vector<int>>> result;
        for (size_t i = 0; i < fold; i++)
        {
            vector<int> subsetIndexes;
            size_t begin = cardNum / fold * i,
                end = (i != fold - 1) ? cardNum / fold * (i + 1) : cardNum;

            for (size_t j = begin; j < end; j++)
                subsetIndexes.push_back(permutation[j]);
                
            result.push_back(Divide(vec, subsetIndexes));
        }

        return result;
    }
}