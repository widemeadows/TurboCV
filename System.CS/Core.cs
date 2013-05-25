using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace System.CS
{
    internal class Mat<T>
    {
        public Mat(int rows, int cols)
        {
            _mat = new T[rows, cols];
        }

        public int Rows
        {
            get
            {
                return _mat.GetLength(1);
            }
        }

        public int Cols
        {
            get
            {
                return _mat.GetLength(0);
            }
        }

        public T this[int i, int j]
        {
            get
            {
                return _mat[i, j];
            }

            set
            {
                _mat[i, j] = value;
            }
        }

        public void Set(T value)
        {
            for (int i = Rows - 1; i >= 0; i--)
                for (int j = Cols - 1; j >= 0; j--)
                    _mat[i, j] = value;
        }

        public Mat<T> Clone()
        {
            Mat<T> dst = new Mat<T>(Rows, Cols);

            for (int i = Rows - 1; i >= 0; i--)
                for (int j = Cols - 1; j >= 0; j--)
                    dst[i, j] = _mat[i, j];

            return dst;
        }

        private T[,] _mat = null;
    }
}
