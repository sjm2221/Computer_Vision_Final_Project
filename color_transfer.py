import numpy as np
import math
import skimage
from skimage import io, color
import matplotlib.pyplot as plt
import argparse

def ReshapeHistogram(Is, It, perc, weight=False, bw=False):
    #convert Is, It to CIELab color space
    Is = color.rgb2lab(Is)
    It = color.rgb2lab(It)
    
    # Check if the image is in black and white
    # If so, copy the first channel over to the second and third channels
    if bw is True:
        Is[:,:,1] = Is[:,:,0].copy()
        Is[:,:,2] = Is[:,:,0].copy()
    
    # Create a variable to store the output
    Io = Is.copy()
    

    # V holds the histogram bin width
    V = 0.3

    # Find the min and max values of source and target
    Is_max = np.max(Is)
    It_max = np.max(It)
    Is_min = np.min(Is)
    It_min = np.min(It)
    # find the minimum value
    I_min = min(Is_min, It_min)

    # Find the number of bins by calculating the range of values and dividing by the bin size
    B = math.ceil((max(Is_max, It_max) - min(Is_min, It_min))/V)
    # Set minimum allowed histogram size
    Bmin = 10

    # compute Smax
    # Smax is the number of scales at which we wish to perform color transfer
    # set the maximum number of scales
    Smax = math.floor(math.log2(B/Bmin))

    
    # Go through each channel of the images
    for i in range(3):
        Is_c = Is[:,:,i]
        It_c = It[:,:,i]
        
        #create histogram
        # initialize the source and target histograms to have the correct number of bins all with value 0
        Hs = [0 for i in range(B)]
        Ht = [0 for i in range(B)]
        
        # go through each pixel of the current channel of source
        for k in range(Is_c.shape[0]):
            for m in range(Is_c.shape[1]):
                # find which bin the current pixel belongs in
                bin_idx = math.floor((Is_c[k, m] - I_min)/V)
                # update the count of the bin of source histogram
                Hs[bin_idx] += 1

        # go through each pixel of the current channel of target
        for k in range(It_c.shape[0]):
            for m in range(It_c.shape[1]):
                # find which bin the current pixel belongs in
                bin_idx = math.floor((It_c[k, m] - I_min)/V)
                # update the count of the bin of source histogram
                Ht[bin_idx] += 1
      
        Hsk = np.array(Hs)
        Htk = np.array(Ht)
        
        #print('sums: ', np.sum(Hsk), np.sum(Htk))
        
        # Normalize the counts of target to be similar to those of source
        Htk = Htk * (np.sum(Hsk)/np.sum(Htk))
        Ht_ = Htk.copy()
        
        #print('sums: ', np.sum(Hsk), np.sum(Htk))
        Hok = np.array(Hs)
        
        # Go through each scale
        for k in range(1, int(perc*Smax)+1):
            print(i, k)
            
            #smooth histogram
            scale = math.pow(2, Smax-k)
            Hsk_temp = Hsk.copy()
            
            # Smooth the histograms given the current scale
            for k_ in range(len(Hs)):
                r_max = min(k_ + int(scale/2)+1, len(Hs))
                r_min = max(k_ - int(scale/2), 0)
                s = np.mean(Hsk[r_min:r_max])

                Hsk_temp[k_] = s
                t = np.mean(Ht_[r_min:r_max])
                Htk[k_] = t
            Hsk = Hsk_temp
            
            # Blend the source and target histograms based on the current scale
            # Lower scales will have more of the source histogram, while higher scales
            # will have more of the target histogram
            if weight is True:
                Htk = Htk*(k/Smax) + Hsk*(1-k/Smax)
            
            # Find the peaks using minima and maxima
            Rmint = FindPeaks(Htk)
            Rmaxt = FindPeaks_max(Htk)
            
            Hsk_ = Hsk.copy()            

            # m_max stores an offest for the maxima list
            m_max = 0
            # Go through each of the maxima found
            for m in range(len(Rmaxt)):
                # Check if the current max is less than the smallest minima
                if Rmaxt[m] < Rmint[0]:
                    m_max += 1
                else:
                    break

            # Go through the minima to reshape regions based on target peaks
            for m in range(len(Rmint)-1):
                # If we have already looked at all the peaks as described by the maxima,
                # exit this loop
                if m_max == len(Rmaxt):
                    break
                
                # Check if the max is in the range of the two minimum
                if Rmaxt[m_max] in range(Rmint[m]+1, Rmint[m+1]):
                    # Reshape the source histogram
                    Hsk_[Rmint[m]:Rmint[m+1]] = RegionTransfer(Hsk[Rmint[m]:Rmint[m+1]], Htk[Rmint[m]:Rmint[m+1]], k/Smax)
                    m_max += 1
                    
            #Hsk_ = Hsk_ * (np.sum(Htk)/np.sum(Hsk_))
            
            # Perform the same process as above, but use regions based on the source histogram instead of the target histogram
            # Each peak of the source, bound by two minima, will be a region of interest
            # The region of interest of the source is then reshpaed according to the corresponding region in the target
            Rmins = FindPeaks(Hsk_)
            Rmaxs = FindPeaks_max(Hsk_)
            m_max = 0
            for m in range(len(Rmaxs)):
                if Rmaxs[m] < Rmins[0]:
                    m_max += 1
                else:
                    break
            Hok = Hsk_.copy()
            for m in range(len(Rmins)-1):
                if m_max == len(Rmaxs):
                    break
                if Rmaxs[m_max] in range(Rmins[m]+1, Rmins[m+1]):
                    Hok[Rmins[m]:Rmins[m+1]] = RegionTransfer(Hsk_[Rmins[m]:Rmins[m+1]], Htk[Rmins[m]:Rmins[m+1]], k/Smax)
                    m_max += 1
                    
            #print('sum: ', np.sum(Htk), np.sum(Hok))            
            #DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs), Htk, I_min, V, i, k)
            DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs), Ht_, I_min, V, i, k)

            # Use the output of this scale as the source for the next scale
            Hsk = Hok * (np.sum(Htk)/np.sum(Hok))
                
        # Update output channel to match the current channel
        Io[:,:,i] = HistMatch(Is_c, I_min, Hs, Hsk, V)
    
    return color.lab2rgb(Io)

# This is the same as the ReshapeHistogram method, except instead of using a bin width V to find a number of bins B,
# we use a set bin size B and, from there, can find the bin width V
def ReshapeHistogram_fixed_bin_size(Is, It, perc, B=400, weight=False, bw=False):
    #convert Is, It to CIELab color space
    #Is = skimage.filters.gaussian(Is)
    #It = skimage.filters.gaussian(It)
    Is = color.rgb2lab(Is)
    It = color.rgb2lab(It)
    if bw is True:
        Is[:,:,1] = Is[:,:,0].copy()
        Is[:,:,2] = Is[:,:,0].copy()
    Io = Is.copy()

    #compute Smax
    #V = 0.3
    Is_max = np.max(Is)
    It_max = np.max(It)
    Is_min = np.min(Is)
    It_min = np.min(It)
    I_max = max(Is_max, It_max)
    #print('total I_max=', I_max)
    
    Bmin = 10
    Smax = math.floor(math.log2(B/Bmin))
    print(Smax)
    
    
    for i in range(3):

        Is_c = Is[:,:,i]
        It_c = It[:,:,i]
        
        Is_max = np.max(Is_c)
        It_max = np.max(It_c)
        Is_min = np.min(Is_c)
        It_min = np.min(It_c)
        I_min = min(Is_min, It_min)
        V = (max(Is_max, It_max) - min(Is_min, It_min))/(float(B-1))
        print('V=%f, Imin=%f, Imax=%f' % (V, I_min, max(Is_max, It_max)))
        
        #create histogram
        Hs = [0 for i in range(B)]
        Ht = [0 for i in range(B)]
        
        for k in range(Is_c.shape[0]):
            for m in range(Is_c.shape[1]):
                bin_idx = math.floor((Is_c[k, m] - I_min)/V)
                Hs[bin_idx] += 1

        for k in range(It_c.shape[0]):
            for m in range(It_c.shape[1]):
                bin_idx = math.floor((It_c[k, m] - I_min)/V)
                Ht[bin_idx] += 1

        Hsk = np.array(Hs)
        Htk = np.array(Ht) 
        #print('sums: ', np.sum(Hsk), np.sum(Htk))
        Htk = Htk * (np.sum(Hsk)/np.sum(Htk)) #
        Ht_ = Htk.copy()
        
        #print('sums: ', np.sum(Hsk), np.sum(Htk))
        Hok = np.array(Hs)
        for k in range(1, int(perc*Smax)+1):
            print(i, k)
            #smooth histogram
            scale = math.pow(2, Smax-k)
            Hsk_temp = Hsk.copy()
            for k_ in range(len(Hs)):
                r_max = min(k_ + int(scale/2)+1, len(Hs))
                r_min = max(k_ - int(scale/2), 0)
                s = np.mean(Hsk[r_min:r_max])
                #print(s)
                Hsk_temp[k_] = s
                t = np.mean(Ht_[r_min:r_max])
                Htk[k_] = t
            Hsk = Hsk_temp

            if weight is True:
                Htk = Htk*(k/Smax) + Hsk*(1-k/Smax)

            Rmint = FindPeaks(Htk)
            Rmaxt = FindPeaks_max(Htk)
            
            #region transfer based on target
            Hsk_ = Hsk.copy()
            print("weight: ", k/Smax)
            m_max = 0
            for m in range(len(Rmaxt)):
                if Rmaxt[m] < Rmint[0]:
                    m_max += 1
                else:
                    break
            for m in range(len(Rmint)-1):
                if m_max == len(Rmaxt):
                    break
                if Rmaxt[m_max] in range(Rmint[m]+1, Rmint[m+1]):                    
                    Hsk_[Rmint[m]:Rmint[m+1]] = RegionTransfer(Hsk[Rmint[m]:Rmint[m+1]], Htk[Rmint[m]:Rmint[m+1]], k/Smax)
                    m_max += 1                   
            #Hsk_ = Hsk_ * (np.sum(Htk)/np.sum(Hsk_))
            
            #region transfer based on source
            Rmins = FindPeaks(Hsk_)
            Rmaxs = FindPeaks_max(Hsk_)
            m_max = 0
            Hok = Hsk_.copy()
            for m in range(len(Rmaxs)):
                if Rmaxs[m] < Rmins[0]:
                    m_max += 1
                else:
                    break
            for m in range(len(Rmins)-1):
                if m_max == len(Rmaxs):
                    break
                if Rmaxs[m_max] in range(Rmins[m]+1, Rmins[m+1]):      
                    #print(Rmins[m]+1, Rmins[m+1])              
                    Hok[Rmins[m]:Rmins[m+1]] = RegionTransfer(Hsk_[Rmins[m]:Rmins[m+1]], Htk[Rmins[m]:Rmins[m+1]], k/Smax)
                    m_max += 1
            #print('sum: ', np.sum(Htk), np.sum(Hok))            
            DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hsk), Htk, I_min, V, i, k)
            #DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs), Ht_, I_min, V, i, k)
            Hsk = Hok * (np.sum(Htk)/np.sum(Hok))
        Io[:,:,i] = HistMatch(Is_c, I_min, Hs, Hsk, V)
    
    return color.lab2rgb(Io)


#def P(i, j, V, min_I):
#    if j == math.floor((i - min_I)/V):
#        return 1
#    return 0

# Find the minima
def FindPeaks(H):
    # intitialize empty list of minima
    H_ = []
    
    # Find the first derivative of the histogram H using forward differencing
    for i in range(len(H)-1):
        H_.append(H[i+1] - H[i])
    
    # Go through the first derivatives and find the points that cross zero (the sign changes between consecutive first derivatives)
    H_mult = []
    for i in range(len(H_)-1):
        H_mult.append(H_[i]*H_[i+1])

    #Find the second derivative using forward differencing on first derivative list
    H2 = []
    for i in range(len(H_)-1):
        H2.append(H_[i+1] - H_[i])

    # Find points where the function crosses 0 (H_mult <= 0) and is increasing (H2 > 0)
    Hr = np.logical_and(np.array(H2)>0, np.array(H_mult)<=0)

    # Find indices of minima
    Rmin = np.argwhere(Hr == True)
    Rmin = [r[0]+1 for r in Rmin]

    return Rmin

#Find maxima using similar technique as FindPeaks
# Only change is check if H2 < 0 instead of > 0 (meaning the function is decreasing)
def FindPeaks_max(H):
    H_ = [] #gradient
    for i in range(len(H)-1):
        H_.append(H[i+1] - H[i])

    H_mult = [] #multiplied
    for i in range(len(H_)-1):
        H_mult.append(H_[i]*H_[i+1])
    H2 = [] #second derivative
    for i in range(len(H_)-1):
        H2.append(H_[i+1] - H_[i])
            
    Hr = np.logical_and(np.array(H2)<0, np.array(H_mult)<0)

    Rmax = np.argwhere(Hr == True)
    Rmax = [r[0]+1 for r in Rmax]
    return Rmax


def RegionTransfer(Hs, Ht, wt, weight=False):
    #ws = 1.0
    #wt = 1.0
    Ho = []
    
    # Check for division by 0
    # If so, we ignore the standard deviations of this region and just use 1 in the reshaping equation
    d = np.std(Ht)/(np.std(Hs))
    if np.std(Hs)== 0:
        #d = wt/ws
        d = 1.0

    # Reshape the region using the meand and standard deviation of source and target
    for hs in Hs:
        hs_new = ((hs - np.mean(Hs))*(d) + np.mean(Ht))
        #hs_diff = hs_new - hs
        #Ho.append (hs + hs_diff*wt)
        Ho.append(hs_new)
    
    Ho = np.array(Ho)
    #return reshaped region with negative values clamped to 0
    return np.where(Ho < 0, 0, Ho)


def HistMatch(Is, Imin, Hs, Ho, V=1):
    # Normalize the count of output to be within range of source histogram
    Ho_ = Ho * np.sum(Hs)/np.sum(Ho)
    #print('histmatch sums:', np.sum(Hs), np.sum(Ho))
    
    # Create cumulative histograms from source and target histograms
    Cs = np.cumsum(Hs)
    Co = np.cumsum(Ho_)
    #create inverse fn
    Co_inv = np.repeat(len(Cs)-1, len(Cs))
    
    # for each bin in the source histogram...
    for i in range(len(Cs)):
        # Get the count of current bin
        c = Cs[i]        
        if c < Co[0]:
            Co_inv[i]=0
        else:
            # Look through each bin of the output histogram
            for j in range(1, len(Co)):
                # find the bin that matches the value of c
                if c >= Co[j-1] and c <Co[j]:
                    Co_inv[i] = j
                    break
    Io = Is.copy()

    #bin_idx = 0    
    # Go through each pixel of the output
    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
            # find the bin that the pixel belongs to
            bin_idx = Co_inv[int((Is[i, j] - Imin)/V)]
            # set pixel value
            Io[i, j] = Imin + (bin_idx)*V

    return Io

def DrawHistogram(H, Hs, Ht, Imin, V=1, ic=0, scale=0):
    font = {'fontname':'Times New Roman'}
    plt.rc('font',family='Times New Roman')

    plt.figure()
    plt.xlabel('v', **font)

    v = []
    for i in range(len(Hs)):
        v.append(Imin + (i)*V)
    
    if Hs is not None:
        plt.plot(v, Hs)
    if Ht is not None:
        plt.plot(v, Ht)
    if H is not None:
        plt.plot(v, H)
    plt.legend(['source', 'target', 'output'])
    #plt.show()

    plt.savefig('histogram/histogram_%d_%d.png'%(ic, scale))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=1.0, help='level of scales to use (0.2, 0.4, 0.6, 0.8, 1.0)')
    parser.add_argument('--source', type=str, default='ballerina.jpg', help='path to source image')
    parser.add_argument('--target', type=str, default='sunset.jpg', help='path to target image')
    parser.add_argument('--output', type=str, default='output.jpg', help='path to save image')
    args = parser.parse_args()

    file1 = args.source
    file2 = args.target
    
    I_s = io.imread(file1)
    I_t = io.imread(file2)
    I_o = ReshapeHistogram(I_s, I_t, perc=args.scale, weight=True, bw=False)
    #I_o = ReshapeHistogram_fixed_bin_size(I_s, I_t, perc=args.scale, weight=True, bw=False)

    io.imsave(args.output, I_o)

    
    return

if __name__ == '__main__':
    main()
