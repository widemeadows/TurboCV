using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace ExperimentCS
{
    /// <summary>
    /// ImageBox.xaml 的交互逻辑
    /// </summary>
    public partial class ImageBox : Window
    {
        public ImageBox(BitmapSource bmp)
        {
            InitializeComponent();

            this.Height = bmp.PixelHeight;
            this.Width = bmp.PixelWidth;
            this.image.Source = bmp;
        }
    }
}
