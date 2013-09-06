#include "../System/System.h"
#include "../System.XML/System.XML.h"
#include "Solver.hpp"
#include <map>
#include <algorithm>
using namespace TurboCV::System::XML;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace Image
    {
        //////////////////////////////////////////////////////////////////////////
        // APIs for Configurations
        //////////////////////////////////////////////////////////////////////////

        map<TString, TString> LoadConfiguration(const TString& configFilePath, const TString& featureName)
        {
            TiXmlDocument doc((const char*)configFilePath);
            doc.LoadFile();

            map<TString, TString> params;

            try
            {
                const TiXmlNode* configs = doc.RootElement()->FirstChild();
                const TiXmlElement* config = configs->FirstChild((const char*)featureName)->ToElement();

                const TiXmlAttribute* attr = config->FirstAttribute();
                params[attr->Name()] = attr->Value();

                while ((attr = attr->Next()) != NULL)
                    params[attr->Name()] = attr->Value();
            }
            catch (string&) 
            {
                return map<TString, TString>();
            }

            return params;
        }

        double GetDoubleValue(
            const map<TString, TString>& params, 
            const TString& paramName, 
            const double defaultValue)
        {
            map<TString, TString>::const_iterator itr = params.find(paramName);

            if (itr == params.end())
                return defaultValue;
            else
                return Double::Parse(itr->second);
        }

        ArrayList<double> GetDoubleList(
            const map<TString, TString>& params, 
            const TString& paramName, 
            const ArrayList<double>& defaultValue)
        {
            map<TString, TString>::const_iterator itr = params.find(paramName);

            if (itr == params.end())
                return defaultValue;
            else
            {
                ArrayList<TString> tokens = itr->second.Split(" ,");
                ArrayList<double> values(tokens.Count());

                for (int i = tokens.Count() - 1; i >= 0; i--)
                    values[i] = Double::Parse(tokens[i]);

                return values;
            }
        }

        //////////////////////////////////////////////////////////////////////////
        // APIs for Datasets
        //////////////////////////////////////////////////////////////////////////

        Group<ArrayList<TString>, ArrayList<int>> LoadDataset(const TString& datasetPath)
        {
            DirectoryInfo imageSetInfo(datasetPath);

            ArrayList<TString> classInfos = imageSetInfo.GetDirectories();
            sort(classInfos.begin(), classInfos.end());

            ArrayList<TString> imagePaths;
            ArrayList<int> imageLabels;
            for (int i = 0; i < classInfos.Count(); i++)
            {
                ArrayList<TString> fileInfos = DirectoryInfo(classInfos[i]).GetFiles();
                sort(fileInfos.begin(), fileInfos.end());

                for (int j = 0; j < fileInfos.Count(); j++)
                {
                    imagePaths.Add(fileInfos[j]);
                    imageLabels.Add(i + 1);
                }
            }

            return CreateGroup(imagePaths, imageLabels);
        }

        ArrayList<ArrayList<size_t>> SplitDatasetRandomly(const ArrayList<int> labels, int nFold)
        {
            return RandomSplit(labels.Count(), nFold);
        }

        ArrayList<ArrayList<size_t>> SplitDatasetEqually(const ArrayList<int> labels, int nFold)
        {
            int nItem = labels.Count();
            ArrayList<ArrayList<size_t>> result(nFold);

            ArrayList<Group<int, int>> labelAndIdx(nItem);
            for (int i = 0; i < nItem; i++)
                labelAndIdx[i] = CreateGroup(labels[i], i);
            sort(labelAndIdx.begin(), labelAndIdx.end());

            int begin = 0, end = 0;
            while (end <= nItem)
            {
                if (end == nItem || labelAndIdx[end].Item1() != labelAndIdx[begin].Item1())
                {
                    int nCategory = end - begin;

                    ArrayList<ArrayList<size_t>> pass = RandomSplit(nCategory, nFold);
                    for (int i = 0; i < nFold; i++)
                        for (int j = 0; j < pass[i].Count(); j++)
                            result[i].Add(labelAndIdx[begin + pass[i][j]].Item2());

                    begin = end;
                }

                end++;
            }

            return result;
        }
    }
}
}