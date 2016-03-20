#include <contrib\contrib.hpp>

//
//
//namespace TurboCV
//{
//namespace System
//{
//    namespace ML
//    {
//        class LDAOperator
//        {
//        public:
//            static vector<Histogram> ComputeLDA(const vector<Histogram>& data,
//                const vector<int>& labels, int componentNum)
//            {
//                Mat convert(data.size(), data[0].size(), CV_64F);
//                for (int i = 0; i < data.size(); i++)
//                    for (int j = 0; j < data[i].size(); j++)
//                        convert.at<double>(i, j) = data[i][j];
//
//                LDA lda(data, labels, componentNum);
//
//                Mat result = lda.project(data);
//
//                vector<Histogram> tmp(result.rows);
//                for (int i = 0; i < result.rows; i++)
//                    for (int j = 0; j < result.cols; j++)
//                        tmp[i].push_back(result.at<double>(i, j));
//
//                return tmp;
//            }
//
//            static pair<vector<Histogram>, vector<Histogram>> ComputeLDA(
//                const vector<Histogram>& trainingData, const vector<int>& labels,
//                int componentNum, const vector<Histogram>& evaluationData)
//            {
//                Mat convert1(trainingData.size(), trainingData[0].size(), CV_64F),
//                    convert2(evaluationData.size(), evaluationData[0].size(), CV_64F);
//                for (int i = 0; i < trainingData.size(); i++)
//                    for (int j = 0; j < trainingData[i].size(); j++)
//                        convert1.at<double>(i, j) = trainingData[i][j];
//                for (int i = 0; i < evaluationData.size(); i++)
//                    for (int j = 0; j < evaluationData[i].size(); j++)
//                        convert2.at<double>(i, j) = evaluationData[i][j];
//
//                LDA lda(convert1, labels, componentNum);
//
//                Mat result1 = lda.project(convert1);
//                Mat result2 = lda.project(convert2);
//
//                vector<Histogram> tmp1(result1.rows), tmp2(result2.rows);
//                for (int i = 0; i < result1.rows; i++)
//                    for (int j = 0; j < result1.cols; j++)
//                        tmp1[i].push_back(result1.at<double>(i, j));
//                for (int i = 0; i < result2.rows; i++)
//                    for (int j = 0; j < result2.cols; j++)
//                        tmp2[i].push_back(result2.at<double>(i, j));
//
//                return make_pair(tmp1, tmp2);
//            }
//        };
//    }
//}
//}