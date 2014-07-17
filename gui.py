#!/usr/bin/python

import sys
from PyQt4 import QtGui as qtg
from PyQt4 import QtCore as qtc

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import ExampleAlg
import numpy as np
import collections
import types

FPATH = "_05.wav"

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w')
        self.axes = self.fig.add_subplot(1,1,1)

        #Equivalent to hold(off) in MATLAB, i.e. each plot is fresh
        #without showing old data
        self.axes.hold(False)

        #Plot color order. For more information
        #see http://matplotlib.sourceforge.net/api/axes_api.html#matplotlib.axes.Axes.plot
        self.colors = ['b', 'r', 'g', 'c', 'm']

        #Zoom box color information
        self.zoom_color = 'y'

        #State variables must be here in order to retain state between
        #plot calls
        self.alg = ExampleAlg.ExampleAlg(FPATH)
        self.zoom = {"x":[],
                     "y":[]}

        #State flag to see if zooming mode is active. Set in the left_pressed
        #when the event for left_held is connected, then released when
        #left_released is called
        self.zooming = None

        #Zoom_box holds the x and y values for current zoom box when
        #self.zooming == True
        self.zoom_box = {"x":{},
                         "y":{}}
        self.zoom_box["x"] = {"data_coords":[],
                              "axes_coords":[]}
        self.zoom_box["y"] = {"data_coords":[],
                              "axes_coords":[]}

        #State storage for the current cursor position in data coordinates
        self.cursor_data = {}
        self.cursor_data["x"] = 0
        self.cursor_data["y"] = 0

        #Setting to hold number of channels coming from algorithm
        self.num_chans = 0
        #Array which wil hold T/F values for which channels to display
        self.display_chans = []

        #Maximum zoom is 0, x_max and 0, y_max for the x and y axes
        self.x_max = 0
        self.y_max = 0
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

class DynamicMplCanvas(MplCanvas):
    """ A canvas that updates itself every X seconds with a new plot. """
    def __init__(self, *args, **kwargs):
        #Initialize parent
        MplCanvas.__init__(self, *args, **kwargs)

        #Set initial plot and initial states
        self.compute_initial_figure()

        #Create dynamic canvas and start plotting, set timer for graph updates
        timer = qtc.QTimer(self)
        qtc.QObject.connect(timer,qtc.SIGNAL("timeout()"),self.update_figure)
        X = 750 #in milliseconds
        timer.start(X)

    def draw_figure(self, data):
        """ Handles all the drawing code that is shared by the initial plotting
            and the dynamic plotting. """
        #Link channels in order with the colors list presented by self.colors.
        #Note that if data is shorter than colors list, the end channels will
        #"disappear"
        #TODO: Add skip list to silence channels during runtime
        display = self.display_chans
        colors = self.colors
        args = []
        for tg, ch, col in zip(display, data, colors):
            if tg == True:
                args.append(ch)
                args.append(col)

        self.axes.plot(*args)

        #xs and ys hold the state values for what we want the zoom to be
        self.axes.set_xlim(self.zoom["x"][0], self.zoom["x"][1])
        self.axes.set_ylim(self.zoom["y"][0], self.zoom["y"][1])

        #Display X axes in units of frequency, but we want to leave all the state storage and algorithmic stuff in bin units
        #self.axes.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x*float(self.alg.framerate)/self.alg.fftlen))

        #Draw lines for zooming rectangle, with one axis being in data coords
        #and the other being in axes coords - see
        #http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.axhspan
        if self.zooming != None:
            try:
                self.axes.axhspan(self.zoom_box["y"]["data_coords"][0],
                                  self.zoom_box["y"]["data_coords"][1],
                                  self.zoom_box["x"]["axes_coords"][0],
                                  self.zoom_box["x"]["axes_coords"][1],
                                  color=self.zoom_color,
                                  alpha=.5)
                self.axes.axvspan(self.zoom_box["x"]["data_coords"][0],
                                  self.zoom_box["x"]["data_coords"][1],
                                  self.zoom_box["y"]["axes_coords"][0],
                                  self.zoom_box["y"]["axes_coords"][1],
                                  color=self.zoom_color,
                                  alpha=.5)
            except IndexError:
                #Ignore indexing exceptions - sometimes zoom_box has not been
                #filled when plot is called
                pass

        #Create text in the bottom left that show the data coordinates which the
        #mouse is currently hovering over
        x = "%s" % float("%.2f" % self.cursor_data["x"])
        y = "%s" % float("%.2f" % self.cursor_data["y"])
        self.axes.text(-.1, -.1, "x="+x+"  y="+y, transform = self.axes.transAxes)
        self.draw()

    def compute_initial_figure(self):
         """Initialize figure and set maximum X and maximum Y"""
         #Get first result from algorithm
         self.alg.start()
         res = self.alg.run()

         #Get number of chans in order to set up toggle boxes
         self.num_chans = len(res)
         self.display_chans = [False for i in range(self.num_chans)]

         #Find maximum value of all channels, excluding DC term ([1:])
         max_max = max(map(lambda x: max(x[1:]), res))

         #Find length of longest channel
         self.x_max = max(map(len, res))

         #1.05 is a cushion value so that we can see all of the data at
         #farthest zoom out
         self.y_max = 1.05*max_max

         #Set zoom state to maximum zoom out
         self.zoom["x"] = [0, self.x_max]
         self.zoom["y"] = [0, self.y_max]
         self.axes.set_xlim(self.zoom["x"][0], self.zoom["x"][1])
         self.axes.set_ylim(self.zoom["y"][0], self.zoom["y"][1])

         self.draw_figure(res)

    def update_figure(self):
        """ Plot the new data, and set zoom levels to current state values. """
        #Get values for next algorithm process
        res = self.alg.run()

        #Plot new data using configured color scheme
        self.draw_figure(res)

class AlgGui(qtg.QWidget):
    """ Main GUI class, defines mouse and keyboard control functionality. """
    #To see a tutorial on using the transforms...
    #http://matplotlib.sourceforge.net/users/transforms_tutorial.html
    def __init__(self):
        qtg.QWidget.__init__(self)
        self.graph = DynamicMplCanvas(self, width=10, height=10, dpi=100)

        #Storage for click coordinates during click state
        self.coords = {"x":[],
                       "y":[]}
        self.initUI()

    def genEditFunction(self, key, le, mn, mx):
        """ Generator function for making a specific textChanged function
            in order to connect to a QLineEdit box. Only works for integer
            inputs to QLineEdit box. """
        def textChanged(string):
            #Check that le is between mn and mx
            pos = 0
            v = qtg.QIntValidator(mn, mx, le)
            le.setValidator(v)

            #Bounds checking
            if v.validate(string, pos) == qtg.QValidator.Invalid:
               value = self.graph.alg.adjustable_params[key]["current_value"]
               le.setText(str(value))
               print("Input of " + str(string) + " is outside range " + str(mn) + "," + str(mx))
            else:
                try:
                    self.graph.alg.adjustable_params[key]["current_value"] = int(string)
                except ValueError:
                    #Do this to suppress printing of error when line is blank
                    pass
        return textChanged

    def genIdleFunction(self, key, le):
        """ Generator for a super simple test of box contents. """
        def editingFinished():
            if len(le.text()) < 1:
                self.graph.alg.adjustable_params[key]["min_value"]
                le.setText(str(value))
        return editingFinished

    def genSliderFunction(self, key, le, mn, mx):
        """ Generator function for making the value changed function for a particular slider """
        def valueChanged(value):
            res = value*mx/100 if value*mx/100 > mn else mn
            le.setText(str(res))
            self.graph.alg.adjustable_params[key]["current_value"] = res
        return valueChanged

    def addSliders(self, widgets):
        """ Function to add arbitrary number of sliders to the display """
        for key in self.graph.alg.adjustable_params.keys():
            #Add a label to the widgets dict
            widgets[str(key) + "_label"] = qtg.QLabel(str(key))

            #Get data extents for bounds checking
            mn = self.graph.alg.adjustable_params[key]["min"]
            mx = self.graph.alg.adjustable_params[key]["max"]

            #Create a line edit widget and connect it to the generated
            #textChanged function from the genEditFunction
            le = qtg.QLineEdit(self)
            edit = self.genEditFunction(key, le, mn, mx)
            le.textChanged.connect(edit)

            #Set text to min value if editing finishes as blank...
            #Currently bugged in Ubuntu 11.10
            fin = self.genIdleFunction(key, le)
            le.editingFinished.connect(fin)

            #Set text to default value
            value = self.graph.alg.adjustable_params[key]["current_value"]
            le.setText(str(value))
            widgets[str(key) + "_current_value"] = le

            #Create a slider, connect it to the generated sliderFunction,
            #and add it to the widgets dict
            sld = qtg.QSlider(qtc.Qt.Horizontal, self)
            fn = self.genSliderFunction(key, le, mn, mx)
            sld.valueChanged.connect(fn)
            widgets[str(key) + "_slider"] = sld

            #Add an empty space, so that widgets are better grouped visually
            widgets[str(key) + "_spacer"] = qtg.QLabel(" ")

    def boundsCheck(self, xdata, ydata):
        """Make sure that zoom boundaries are within data window"""
        xdata = self.graph.zoom["x"][0] if xdata < self.graph.zoom["x"][0] else xdata
        xdata = self.graph.zoom["x"][1] if xdata > self.graph.zoom["x"][1] else xdata
        ydata = self.graph.zoom["y"][0] if ydata < self.graph.zoom["y"][0] else ydata
        ydata = self.graph.zoom["y"][1] if ydata > self.graph.zoom["y"][1] else ydata
        return (xdata, ydata)

    def left_pressed(self, event):
        """Record location where the left click started"""
        #Use the transform so we enable the ability to click outside axes,
        #as event.xdata = None if event.inaxes == False
        #Also make sure not to zoom outside data bounds
        if event.button == 1:
            xdata, ydata = self.graph.axes.transData.inverted().transform((event.x, event.y))
            xdata, ydata = self.boundsCheck(xdata, ydata)

            #Add location data to self.coords for storage
            self.coords["x"].append(xdata)
            self.coords["y"].append(ydata)

            #Set the zooming state so it is no longer None
            self.graph.zooming = self.graph.mpl_connect("motion_notify_event", self.left_held)

    def left_held(self, event):
        """Method for use during zoom event"""
        #Get x and y coordinates from data coords where left click started
        x_temp, y_temp = self.graph.axes.transData.transform((self.coords["x"][0], self.coords["y"][0]))

        #Get x and y data points for where the current event is
        x0, y0 = self.graph.axes.transData.inverted().transform((event.x, event.y))
        #Save off data coords
        self.graph.zoom_box["x"]["data_coords"] = sorted([self.coords["x"][0], x0])
        self.graph.zoom_box["y"]["data_coords"] = sorted([self.coords["y"][0], y0])
        #Get axes coordinates for where left click started
        x1, y1 = self.graph.axes.transAxes.inverted().transform((x_temp, y_temp))
        #Get current coordinates of cursor
        x2, y2 = self.graph.axes.transAxes.inverted().transform((event.x, event.y))
        #Make sure the box is always left, right and lower, higher
        self.graph.zoom_box["x"]["axes_coords"] = sorted([x1, x2])
        self.graph.zoom_box["y"]["axes_coords"] = sorted([y1, y2])

    def left_released(self, event):
        """Record location of click release, then update axes state"""
        if event.button == 1:
            #Get data coordinate for event. Use this method because event.x and
            #event.y return None when event.inaxes == None
            xdata, ydata = self.graph.axes.transData.inverted().transform((event.x, event.y))
            xdata, ydata = self.boundsCheck(xdata, ydata)

            #Append release coordinates to the stored value for where left click
            #started.
            self.coords["x"].append(xdata)
            self.coords["y"].append(ydata)
            x_list = self.coords["x"]
            y_list = self.coords["y"]

            #xs and ys hold the zoom state of the plot, so update those
            #TODO: Check that zoom box covers some portion inside the graph
            self.graph.zoom["x"] = sorted(x_list)
            self.graph.zoom["y"] = sorted(y_list)

            #Disconnect event and return zooming flag to None state
            self.graph.mpl_disconnect(self.graph.zooming)
            self.graph.zooming = None

        #Empty out coords, left click is no longer pressed
        self.coords["x"] = []
        self.coords["y"] = []

    def right_pressed(self, event):
        """Zoom out to initial zoom level"""
        if event.button == 3:
            #Zoom to initial state
            self.graph.zoom["x"] = [0, self.graph.x_max]
            self.graph.zoom["y"] = [0, self.graph.y_max]

    def display_cursor_point(self, event):
        """Show the data coordinate where the mouse cursor is hovering"""
        if event.inaxes != None:
            self.graph.cursor_data["x"] = event.xdata
            self.graph.cursor_data["y"] = event.ydata

    def genCheckboxFunction(self, num):
        """Generator for a channel toggle checkboxes. """
        def toggleChannel():
            self.graph.display_chans[num] = not self.graph.display_chans[num]
        return toggleChannel

    def addCheckboxes(self, widgets):
        """Add textboxes to passed in collection."""
        for i in range(self.graph.num_chans):
            cb = qtg.QCheckBox()
            widgets['chan_'+str(i)+'checkbox'] = cb
            fn = self.genCheckboxFunction(i)
            cb.stateChanged.connect(fn)

    def initLayout(self):
        hbox = qtg.QHBoxLayout()

        #Click and drag zooming functions
        self.zoom_start = self.graph.mpl_connect("button_press_event", self.left_pressed)
        self.zoom_end = self.graph.mpl_connect("button_release_event", self.left_released)

        #Undo zoom functions
        self.unzoom = self.graph.mpl_connect("button_press_event", self.right_pressed)

        #Cursor positional display
        self.cursor_pos = self.graph.mpl_connect("motion_notify_event", self.display_cursor_point)

        #Plot graphic
        hbox.addWidget(self.graph)

        vbox = qtg.QVBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox)

        #Top right widgets, pass in widgets dict so sliders can be added
        widgets = collections.OrderedDict()
        self.addSliders(widgets)
        [vbox.addWidget(x) for x in widgets.values()]

        vbox.addStretch(1)

        #Bottom right widgets, pass in checbox_widgets so checkboxes can be added
        vbox.addWidget(qtg.QLabel("Enable Channels 1 - "+str(self.graph.num_chans)))
        hbox_check = qtg.QHBoxLayout()
        checkbox_widgets = collections.OrderedDict()
        self.addCheckboxes(checkbox_widgets)
        [hbox_check.addWidget(x) for x in checkbox_widgets.values()]
        vbox.addLayout(hbox_check)

        self.setLayout(hbox)

    def initUI(self):
        #Set window title to the name of the included algorithm
        self.setWindowTitle(self.graph.alg.__class__.__name__)
        self.initLayout()
        self.show()

if __name__ == "__main__":
    app = qtg.QApplication(sys.argv)
    g = AlgGui()
    sys.exit(app.exec_())
