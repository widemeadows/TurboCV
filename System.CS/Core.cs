using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Turbo.System.CS
{
    public struct Size
    {
        public Size(int h, int w)
        {
            _height = h;
            _width = w;
        }

        public int Height
        {
            get
            {
                return _height;
            }
            set
            {
                _height = value;
            }
        }

        public int Width
        {
            get
            {
                return _width;
            }
            set
            {
                _width = value;
            }
        }

        private int _height, _width;
    }

    public class Mat<T> where T: IComparable<T>
    {
        public Mat(int rows, int cols)
        {
            _mat = new T[rows, cols];
            _size = new Size(rows, cols);
        }
        
        public Mat(Size size)
        {
            _mat = new T[size.Height, size.Width];
            _size = size;
        }

        public Mat(T[,] data)
        {
            _mat = (T[,])data.Clone();
            _size = new Size(_mat.GetLength(1), _mat.GetLength(0));
        }

        public int Rows
        {
            get
            {
                return _size.Height;
            }
        }

        public int Cols
        {
            get
            {
                return _size.Width;
            }
        }

        public Size Size
        {
            get
            {
                return _size;
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

        public Mat<T> Max(T value)
        {
            Mat<T> dst = Clone();

            for (int i = _size.Height - 1; i >= 0; i--)
                for (int j = _size.Width - 1; j >= 0; j--)
                    dst[i, j] = dst[i, j].CompareTo(value) > 0 ? dst[i, j] : value;

            return dst;
        }

        public void Set(T value)
        {
            for (int i = _size.Height - 1; i >= 0; i--)
                for (int j = _size.Width - 1; j >= 0; j--)
                    _mat[i, j] = value;
        }

        public Mat<T> Clone()
        {
            return new Mat<T>(_mat);
        }

        private T[,] _mat = null;
        private Size _size;
    }

    public struct Point
    {
        public Point(int x, int y)
        {
            _x = x;
            _y = y;
        }

        public int X
        {
            get
            {
                return _x;
            }
            set
            {
                _x = value;
            }
        }

        public int Y
        {
            get
            {
                return _y;
            }
            set
            {
                _y = value;
            }
        }

        private int _x, _y;
    }
}
