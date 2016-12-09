# interactive-counting
Demo code and GUI for "Interactive Object Counting"

This is a MATLAB implementation of the detector described in [1].

This package contains a MATLAB GUI and a sample image that allows the teting of the interactive counting framework.

## Dependencies
 
* [vl_feat MATLAB library](http://www.vlfeat.org/) 
* [Peter Kovesi's MATLAB library](http://www.peterkovesi.com/matlabfns/index.html)
* [export_fig tool](https://uk.mathworks.com/matlabcentral/fileexchange/23629-export-fig) 
* (Optional) [demo images](http://www.robots.ox.ac.uk/~vgg/software/interactive_counting/images.zip)

This code has been tested using VL_feat 0.9.14-0.9.20, 
under Ubuntu11.04-14.04 and Windows7-10, with MATLAB2011-2015.

## Usage

To try the code, do the following:

* Install and setup the dependencies
* Run `InteractiveCountingGUI.m`
* Click *File-->Open* and load an image from the demo images
* Click the *ROI* button and select and area of the image containing the object of interest
* Click the *dots* button and place dots on each instace of the object inside the selected ROI
* Click the *diameter* button and place a line on top of an instance that is roughly its diameter
* Click Process

There is currently not a written tutorial on how to use the interface, but it should be easy to find out; [this video](https://www.youtube.com/watch?v=_NwY4fjEW3A&feature=youtu.be) might help: 

The ROI and Diameter tools require a first click on the image to become active.
The Diameter tool should only be used once.

## Issues

If you are running on linux 64-bits, it should work right away. 
Nevertheless, it is possible that there are issues with some matlab versions when it comes to the drawing tools or the plots.
If you are running on a different OS, let me know and I'll tell you what you need to compile.

Please note that this demo was done for the single image scenario. 

## Relevant publications

* [1] C. Arteta, V. Lempitsky, J. A. Noble, A. Zisserman
Interactive Object Counting, ECCV 2014.

## License

Copyright (C) 2014-2016 by Carlos Arteta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

