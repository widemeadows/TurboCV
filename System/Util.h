#pragma once

#include <vector>
#include <cassert>
using namespace std;

namespace System
{
    inline vector<int> RandomPermutate(int cardNum, int pickUpNum)
    {
        assert(cardNum >= pickUpNum);
        vector<int> result;

        int* cards = new int[cardNum];
        for (int i = 0; i < cardNum; i++)
            cards[i] = i;

        for (int i = 0; i < pickUpNum; i++)
        {
            int index = (double)rand() * (cardNum - i - 1) / RAND_MAX + i;
            assert(index >= 0 && index < cardNum);
            swap(cards[i], cards[index]);
        }

        for (int i = 0; i < pickUpNum; i++)
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
        int cardNum = vec.size(), counter = 0;

        for (int i = 0; i < cardNum; i++)
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
    inline tuple<vector<T>, vector<T>, vector<int>> Divide(
        const vector<T>& vec, const vector<int>& pickUpIndexes)
    {
        vector<int> indexes = pickUpIndexes;
        sort(indexes.begin(), indexes.end());

        vector<T> pickUps, others;
        int cardNum = vec.size(), counter = 0;

        for (int i = 0; i < cardNum; i++)
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
    inline vector<T> RandomPickUp(const vector<T>& vec, int pickUpNum)
    {
        int cardNum = vec.size();
        assert(cardNum >= pickUpNum);

        return PickUp(vec, RandomPermutate(cardNum, pickUpNum));
    }

    template<typename T>
    inline vector<tuple<vector<T>, vector<T>, vector<int>>> RandomSplit(
        const vector<T>& vec, int fold)
    {
        int cardNum = vec.size();
        assert(cardNum >= fold);

        vector<int> permutation = RandomPermutate(cardNum, cardNum);

        vector<tuple<vector<T>, vector<T>, vector<int>>> result;
        for (int i = 0; i < fold; i++)
        {
            vector<int> subsetIndexes;
            int begin = cardNum / fold * i,
                end = (i != fold - 1) ? cardNum / fold * (i + 1) : cardNum;

            for (int j = begin; j < end; j++)
                subsetIndexes.push_back(permutation[j]);
                
            result.push_back(Divide(vec, subsetIndexes));
        }

        return result;
    }
}