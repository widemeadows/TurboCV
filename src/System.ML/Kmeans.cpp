#define HAVE_TBB

#include <cv.h>
#include <vector>
using namespace cv;
using namespace std;

namespace TurboCV
{
namespace System
{
    namespace ML
    {
        static void generateRandomCenter(const vector<Vec2f>& box, float* center, RNG& rng)
        {
            size_t j, dims = box.size();
            float margin = 1.f/dims;
            for( j = 0; j < dims; j++ )
                center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
        }

        class KMeansPPDistanceComputer
        {
        public:
            KMeansPPDistanceComputer( float *_tdist2,
                const float *_data,
                const float *_dist,
                int _dims,
                size_t _step,
                size_t _stepci )
                : tdist2(_tdist2),
                data(_data),
                dist(_dist),
                dims(_dims),
                step(_step),
                stepci(_stepci) { }

            void operator()( const cv::BlockedRange& range ) const
            {
                const int begin = range.begin();
                const int end = range.end();

                for ( int i = begin; i<end; i++ )
                {
                    tdist2[i] = std::min(normL1_(data + step*i, data + stepci, dims), dist[i]);
                }
            }

        private:
            KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

            float *tdist2;
            const float *data;
            const float *dist;
            const int dims;
            const size_t step;
            const size_t stepci;
        };

        /*
        k-means center initialization using the following algorithm:
        Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
        */
        static void generateCentersPP(const Mat& _data, Mat& _out_centers,
            int K, RNG& rng, int trials)
        {
            int i, j, k, dims = _data.cols, N = _data.rows;
            const float* data = _data.ptr<float>(0);
            size_t step = _data.step/sizeof(data[0]);
            vector<int> _centers(K);
            int* centers = &_centers[0];
            vector<float> _dist(N*3);
            float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
            double sum0 = 0;

            centers[0] = (unsigned)rng % N;

            for( i = 0; i < N; i++ )
            {
                dist[i] = normL1_(data + step*i, data + step*centers[0], dims);
                sum0 += dist[i];
            }

            for( k = 1; k < K; k++ )
            {
                double bestSum = DBL_MAX;
                int bestCenter = -1;

                for( j = 0; j < trials; j++ )
                {
                    double p = (double)rng*sum0, s = 0;
                    for( i = 0; i < N-1; i++ )
                        if( (p -= dist[i]) <= 0 )
                            break;
                    int ci = i;

                    parallel_for(BlockedRange(0, N),
                        KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
                    for( i = 0; i < N; i++ )
                    {
                        s += tdist2[i];
                    }

                    if( s < bestSum )
                    {
                        bestSum = s;
                        bestCenter = ci;
                        std::swap(tdist, tdist2);
                    }
                }
                centers[k] = bestCenter;
                sum0 = bestSum;
                std::swap(dist, tdist);
            }

            for( k = 0; k < K; k++ )
            {
                const float* src = data + step*centers[k];
                float* dst = _out_centers.ptr<float>(k);
                for( j = 0; j < dims; j++ )
                    dst[j] = src[j];
            }
        }

        class KMeansDistanceComputer
        {
        public:
            KMeansDistanceComputer( double *_distances,
                int *_labels,
                const Mat& _data,
                const Mat& _centers )
                : distances(_distances),
                labels(_labels),
                data(_data),
                centers(_centers)
            {
            }

            void operator()( const BlockedRange& range ) const
            {
                const int begin = range.begin();
                const int end = range.end();
                const int K = centers.rows;
                const int dims = centers.cols;

                const float *sample;
                for( int i = begin; i<end; ++i)
                {
                    sample = data.ptr<float>(i);
                    int k_best = 0;
                    double min_dist = DBL_MAX;

                    for( int k = 0; k < K; k++ )
                    {
                        const float* center = centers.ptr<float>(k);
                        const double dist = normL1_(sample, center, dims);

                        if( min_dist > dist )
                        {
                            min_dist = dist;
                            k_best = k;
                        }
                    }

                    distances[i] = min_dist;
                    labels[i] = k_best;
                }
            }

        private:
            KMeansDistanceComputer& operator=(const KMeansDistanceComputer&); // to quiet MSVC

            double *distances;
            int *labels;
            const Mat& data;
            const Mat& centers;
        };

        double Kmeans( InputArray _data, int K,
            InputOutputArray _bestLabels,
            TermCriteria criteria, int attempts,
            int flags, OutputArray _centers )
        {
            const int SPP_TRIALS = 3;
            Mat data = _data.getMat();
            bool isrow = data.rows == 1 && data.channels() > 1;
            int N = !isrow ? data.rows : data.cols;
            int dims = (!isrow ? data.cols : 1)*data.channels();
            int type = data.depth();

            attempts = std::max(attempts, 1);
            CV_Assert( data.dims <= 2 && type == CV_32F && K > 0 );
            CV_Assert( N >= K );

            _bestLabels.create(N, 1, CV_32S, -1, true);

            Mat _labels, best_labels = _bestLabels.getMat();
            if( flags & CV_KMEANS_USE_INITIAL_LABELS )
            {
                CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                    best_labels.cols*best_labels.rows == N &&
                    best_labels.type() == CV_32S &&
                    best_labels.isContinuous());
                best_labels.copyTo(_labels);
            }
            else
            {
                if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
                    best_labels.cols*best_labels.rows == N &&
                    best_labels.type() == CV_32S &&
                    best_labels.isContinuous()))
                    best_labels.create(N, 1, CV_32S);
                _labels.create(best_labels.size(), best_labels.type());
            }
            int* labels = _labels.ptr<int>();

            Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
            vector<int> counters(K);
            vector<Vec2f> _box(dims);
            Vec2f* box = &_box[0];
            double best_compactness = DBL_MAX, compactness = 0;
            RNG& rng = theRNG();
            int a, iter, i, j, k;

            if( criteria.type & TermCriteria::EPS )
                criteria.epsilon = std::max(criteria.epsilon, 0.);
            else
                criteria.epsilon = FLT_EPSILON;
            criteria.epsilon *= criteria.epsilon;

            if( criteria.type & TermCriteria::COUNT )
                criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
            else
                criteria.maxCount = 100;

            if( K == 1 )
            {
                attempts = 1;
                criteria.maxCount = 2;
            }

            const float* sample = data.ptr<float>(0);
            for( j = 0; j < dims; j++ )
                box[j] = Vec2f(sample[j], sample[j]);

            for( i = 1; i < N; i++ )
            {
                sample = data.ptr<float>(i);
                for( j = 0; j < dims; j++ )
                {
                    float v = sample[j];
                    box[j][0] = std::min(box[j][0], v);
                    box[j][1] = std::max(box[j][1], v);
                }
            }

            for( a = 0; a < attempts; a++ )
            {
                double max_center_shift = DBL_MAX;
                for( iter = 0;; )
                {
                    swap(centers, old_centers);

                    if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) )
                    {
                        if( flags & KMEANS_PP_CENTERS )
                            generateCentersPP(data, centers, K, rng, SPP_TRIALS);
                        else
                        {
                            for( k = 0; k < K; k++ )
                                generateRandomCenter(_box, centers.ptr<float>(k), rng);
                        }
                    }
                    else
                    {
                        if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) )
                        {
                            for( i = 0; i < N; i++ )
                                CV_Assert( (unsigned)labels[i] < (unsigned)K );
                        }

                        // compute centers
                        centers = Scalar(0);
                        for( k = 0; k < K; k++ )
                            counters[k] = 0;

                        for( i = 0; i < N; i++ )
                        {
                            sample = data.ptr<float>(i);
                            k = labels[i];
                            float* center = centers.ptr<float>(k);
                            j=0;
                        #if CV_ENABLE_UNROLLED
                            for(; j <= dims - 4; j += 4 )
                            {
                                float t0 = center[j] + sample[j];
                                float t1 = center[j+1] + sample[j+1];

                                center[j] = t0;
                                center[j+1] = t1;

                                t0 = center[j+2] + sample[j+2];
                                t1 = center[j+3] + sample[j+3];

                                center[j+2] = t0;
                                center[j+3] = t1;
                            }
                        #endif
                            for( ; j < dims; j++ )
                                center[j] += sample[j];
                            counters[k]++;
                        }

                        if( iter > 0 )
                            max_center_shift = 0;

                        for( k = 0; k < K; k++ )
                        {
                            if( counters[k] != 0 )
                                continue;

                            // if some cluster appeared to be empty then:
                            //   1. find the biggest cluster
                            //   2. find the farthest from the center point in the biggest cluster
                            //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                            int max_k = 0;
                            for( int k1 = 1; k1 < K; k1++ )
                            {
                                if( counters[max_k] < counters[k1] )
                                    max_k = k1;
                            }

                            double max_dist = 0;
                            int farthest_i = -1;
                            float* new_center = centers.ptr<float>(k);
                            float* old_center = centers.ptr<float>(max_k);
                            float* _old_center = temp.ptr<float>(); // normalized
                            float scale = 1.f/counters[max_k];
                            for( j = 0; j < dims; j++ )
                                _old_center[j] = old_center[j]*scale;

                            for( i = 0; i < N; i++ )
                            {
                                if( labels[i] != max_k )
                                    continue;
                                sample = data.ptr<float>(i);
                                double dist = normL1_(sample, _old_center, dims);

                                if( max_dist <= dist )
                                {
                                    max_dist = dist;
                                    farthest_i = i;
                                }
                            }

                            counters[max_k]--;
                            counters[k]++;
                            labels[farthest_i] = k;
                            sample = data.ptr<float>(farthest_i);

                            for( j = 0; j < dims; j++ )
                            {
                                old_center[j] -= sample[j];
                                new_center[j] += sample[j];
                            }
                        }

                        for( k = 0; k < K; k++ )
                        {
                            float* center = centers.ptr<float>(k);
                            CV_Assert( counters[k] != 0 );

                            float scale = 1.f/counters[k];
                            for( j = 0; j < dims; j++ )
                                center[j] *= scale;

                            if( iter > 0 )
                            {
                                double dist = 0;
                                const float* old_center = old_centers.ptr<float>(k);
                                for( j = 0; j < dims; j++ )
                                {
                                    double t = center[j] - old_center[j];
                                    dist += t*t;
                                }
                                max_center_shift = std::max(max_center_shift, dist);
                            }
                        }
                    }

                    printf("%d %f\n", iter, max_center_shift);
                    if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                        break;

                    // assign labels
                    Mat dists(1, N, CV_64F);
                    double* dist = dists.ptr<double>(0);
                    parallel_for(BlockedRange(0, N),
                        KMeansDistanceComputer(dist, labels, data, centers));
                    compactness = 0;
                    for( i = 0; i < N; i++ )
                    {
                        compactness += dist[i];
                    }
                }

                if( compactness < best_compactness )
                {
                    best_compactness = compactness;
                    if( _centers.needed() )
                        centers.copyTo(_centers);
                    _labels.copyTo(best_labels);
                }
            }

            return best_compactness;
        }
    }
}
}