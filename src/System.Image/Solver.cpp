#include "../System/System.h"
#include "../System.XML/System.XML.h"
#include "Solver.hpp"
#include <string>
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
        // Solver
        //////////////////////////////////////////////////////////////////////////

        TiXmlDocument Solver::LoadConfiguration(const TString& configfileName)
        {
            TiXmlDocument doc((const char*)configfileName);
            doc.LoadFile();

            return doc;
        }

        Group<ArrayList<TString>, ArrayList<int>> Solver::LoadDataset(const TString& datasetPath)
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

        map<TString, TString> Solver::GetConfiguration(const TString& featureName)
        {
            map<TString, TString> params;

            try
            {
                TiXmlNode* configs = doc.RootElement()->FirstChild();
                TiXmlElement* config = configs->FirstChild((const char*)featureName)->ToElement();

                TiXmlAttribute* attr = config->FirstAttribute();
                params[attr->Name()] = attr->Value();

                while ((attr = attr->Next()) != NULL)
                    params[attr->Name()] = attr->Value();
            }
            catch (string& e) 
            {
                return map<TString, TString>();
            }

            return params;
        }

        ArrayList<ArrayList<size_t>> Solver::SplitDatasetRandomly(int nFold)
        {
            return RandomSplit(labels.Count(), nFold);
        }

        ArrayList<ArrayList<size_t>> Solver::SplitDatasetEqually(int nFold)
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