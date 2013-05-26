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
            _rows = rows;
            _cols = cols;
        }

        public Mat(T[,] data)
        {
            _mat = (T[,])data.Clone();
            _rows = _mat.GetLength(1);
            _cols = _mat.GetLength(0);
        }

        public int Rows
        {
            get
            {
                return _rows;
            }
        }

        public int Cols
        {
            get
            {
                return _cols;
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
            for (int i = _rows - 1; i >= 0; i--)
                for (int j = _cols - 1; j >= 0; j--)
                    _mat[i, j] = value;
        }

        public Mat<T> Clone()
        {
            return new Mat<T>(_mat);
        }

        private T[,] _mat = null;
        private int _rows = 0, _cols = 0;
    }

    public struct Point
    {
        public Point(int x, int y)
        {
            X = x;
            Y = y;
        }

        public int X { get; set; }
        public int Y { get; set; }
    }
}
