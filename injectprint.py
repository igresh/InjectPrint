from PIL import Image
import numpy as np
import os
from skimage.measure import block_reduce
from scipy import ndimage


def simp_path (point_arr, weight=0.5):
    """
    Simplify the path that the print-head takes. Useful in cases where standard rasterisation would result in a very inefficent path.
    
    weight : float
        Tuning parameter. Weighting (between 0 and 1) of preference of proximity to current point against preference for previous points.
        Low weights will fill blobs better, high weights will follow lines better.
    """

    point_arr = np.array(point_arr)
    nnm = np.ones(point_arr.shape[0], dtype=bool)

    loc = point_arr[0,:2]
    reordered_point_arr = [point_arr[0]]
    point_arr = np.delete(point_arr, 0, axis=0)
    distances = np.zeros(point_arr.shape[0])

    counter = 0

    xbias = np.array([1, 1])
    
    while np.shape(point_arr)[0] != 0:
        newdist = np.sqrt(np.sum(((loc-point_arr[:,:2])*xbias)**2, axis=1))
        ndq = np.quantile(newdist, 0.05)
        newdist[newdist>ndq] = ndq
        
        distances = weight*newdist + distances*(1-weight)
        minidx = np.argmin(distances)
        loc = point_arr[minidx,:2]
        reordered_point_arr.append(point_arr[minidx])
        point_arr = np.delete(point_arr, minidx, axis=0)
        distances = np.delete(distances, minidx, axis=0)
        counter += 1
        olddist = newdist
        olddist = np.delete(olddist, minidx, axis=0)

    return reordered_point_arr

def image_to_gcode (image_fname, image_width_mm, resolution_mm2=0.5, E_per_mm2=1, reduce_path_travelled=True, anti_drop=True, zero=True):
    """
    
    Inputs
    
    image_fname : string
        file name of image
        
    image_size_mm : float
        width of desired image in millimeters
        
    resolution_mm2 : float
        area corresponding to one injection; sets the spatial resolution. The image will
        be discretised into an image_size_mm/resolution_mm2 array.
        
    E_per_mm2 : float
        Amount to extrude for 1 square millimeter
        
    Returns:
    
    G-code coords : list
    
    """
    
    im = Image.open(image_fname)
    im = im.convert('L')           # Convert image to grayscale
    imarr = np.array(im)
    imarr = imarr/np.max(imarr)    # Normalise image to the [0,1] interval
    imarr[imarr<0.5] = 0           # Threshold image
    imarr[imarr>=0.5] = 1
    imarr = np.logical_not(imarr)  # assumes original image is white-on-black
        
    ppm = imarr.shape[1]/image_width_mm  # pixels per millimeter
    
    slice_size = int(np.sqrt(resolution_mm2)*ppm)
    
    if slice_size < 1:
        slice_size = 1
        print ('warning: desired resolution exceeds that of supplied image')

    y_slices = np.ceil(imarr.shape[0] / slice_size).astype(int)
    x_slices = np.ceil(imarr.shape[1] / slice_size).astype(int)

    resize_y = y_slices*slice_size
    resize_x = x_slices*slice_size

    pad_imarr = np.zeros([resize_y, resize_x])
    pad_imarr[0:imarr.shape[0], 0:imarr.shape[1]] = imarr
    
    point_rep = []

    print (y_slices)
    for y_slice in range(y_slices):
        subpointrep = []
        for x_slice in range(x_slices):
            sel_region = pad_imarr[y_slice*slice_size:(y_slice+1)*slice_size, x_slice*slice_size:(x_slice+1)*slice_size]
            COM = ndimage.measurements.center_of_mass(sel_region)

            if not np.isnan(COM[0]) and pad_imarr[int(COM[0]+y_slice*slice_size), int(COM[1]+x_slice*slice_size)]!=0:
                normmass = np.sum(sel_region)/(slice_size**2)
                extruded_vol = E_per_mm2*resolution_mm2*normmass
                subpointrep.append([COM[0]+y_slice*slice_size,
                                  COM[1]+x_slice*slice_size,
                                  extruded_vol])
                
        if is_odd(y_slice):
            subpointrep = list(reversed(subpointrep))
            
        point_rep += subpointrep

    if reduce_path_travelled:
        point_rep = simp_path(point_rep, weight=0.4)

    # Not quite sure why but we need to flip image along x axis
    point_arr = np.array(point_rep)
    point_arr[:,0] = np.max(point_arr[:,0]) - point_arr[:,0]

    fname = f'{os.path.basename(image_fname).split(".")[0]}.gcode'
    Gcode_writer(point_arr, fname, ppm=ppm, zero=zero, anti_drop=anti_drop)
    
    return point_rep

def is_odd(n):
    if n % 2 == 0:
        return False
    else:
        return True
    
    
def Gcode_writer(point_arr, fname, ppm=1, zero=True, anti_drop=True, movespeed=700, espeed=8, linger=0.8, liftoff=2.5):
    """
    Writes the path created within image_to_gcode to file.
    """
    Gcode_pre = \
f'M107\n\
G92 X0 Y0 Z0 E0  ;set coordinates to 0 \n\
G92 E0\n\
M302 P1\n\
G21	;	set units to millimeters\n\
G90	;	use absolute coordinates\n\
M82	;	use absolute distances for extrusion\n\
G92	E0\n\
G1	F{movespeed} Z{liftoff}\n'


    if anti_drop:
        suckup = 0.02
    else:
        suckup = 0
        
    point_arr = np.array(point_arr)
    point_arr[:,0] = point_arr[:,0]/ppm
    point_arr[:,1] = point_arr[:,1]/ppm
    
    if zero:
        point_arr[:,0] = point_arr[:,0] - np.min(point_arr[:,0])
        point_arr[:,0] = point_arr[:,0] - np.max(point_arr[:,0])/2
        point_arr[:,1] = point_arr[:,1] - np.min(point_arr[:,1])
        point_arr[:,1] = point_arr[:,1] - np.max(point_arr[:,1])/2
        
    e_abs = 0.3

    print (len(point_arr))
    with open(fname, mode='w') as fw:
        fw.write(Gcode_pre)
        y,x,e = point_arr[0]
        fw.write(f'G1	F{movespeed}	X{x}	Y{y}\n')
        fw.write(f'G1	F{10}	E{e_abs}\n')
        fw.write(f'G4 P{4000}\n')
        
        pen_depth = -2.5
        for [y,x,e] in point_arr:
            e_abs += e
            fw.write(f'G1	F{movespeed}	X{x}	Y{y}\n')
            fw.write(f'G1	F{movespeed}	X{x}	Y{y}	Z{pen_depth}\n')
            fw.write(f'G1	F{espeed}	E{e_abs+suckup}\n')
            if linger != 0:
                fw.write(f'G1	F{-60*pen_depth/linger}	X{x}	Y{y}	Z0\n')
            fw.write(f'G1	F{movespeed}	X{x}	Y{y}	Z{liftoff}\n')
            if anti_drop:
                fw.write(f'G1	F{espeed}	E{e_abs-suckup}\n')
                
        fw.write(f'G1	F{movespeed}	X{x}	Y{y}	Z10\n')
        fw.write(f'G1	F{movespeed}	X{0}	Y{0}	Z10\n')
