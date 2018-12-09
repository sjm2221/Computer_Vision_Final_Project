import numpy as np
import math
import skimage
from skimage import io, color
import matplotlib.pyplot as plt

def ReshapeHistogram(Is, It, perc, weight=False, bw=False):
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
    V = 0.3
    Is_max = np.max(Is)
    It_max = np.max(It)
    Is_min = np.min(Is)
    It_min = np.min(It)
    #I = np.concatenate([Is, It])
    B = math.ceil((max(Is_max, It_max) - min(Is_min, It_min))/V)
    Bmin = 10
    Smax = math.floor(math.log2(B/Bmin))
    print(Smax)
    I_min = min(Is_min, It_min)
    
    for i in range(3):
        Is_c = Is[:,:,i]
        It_c = It[:,:,i]
        
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
      
        # Hs_ = Hs.copy()
        # Ht_ = Ht.copy()

        Hsk = np.array(Hs)
        Htk = np.array(Ht) 
        #print('sums: ', np.sum(Hsk), np.sum(Htk))
        Htk = Htk * (np.sum(Hsk)/np.sum(Htk)) #
        Ht_ = Htk.copy()
        
        print('sums: ', np.sum(Hsk), np.sum(Htk))
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
            #weights
            if weight is True:
                Htk = Htk*(k/Smax) + Hsk*(1-k/Smax)
            #
            Rmint = FindPeaks(Htk)
            Rmaxt = FindPeaks_max(Htk)
            
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
                    
            print('sum: ', np.sum(Htk), np.sum(Hok))            
            DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs), Htk, I_min, V, i, k)
            #DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs), Ht_, I_min, V, i, k)
            Hsk = Hok * (np.sum(Htk)/np.sum(Hok))
        Io[:,:,i] = HistMatch(Is_c, I_min, Hs, Hsk, V)
    #print(Io)
    
    return color.lab2rgb(Io)

def ReshapeHistogram_fixed_bin_size(Is, It, perc, B=400, bw=False):
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
        
        print('sums: ', np.sum(Hsk), np.sum(Htk))
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
                #print(Rmaxs[m_max])
                #print(range(Rmins[m]+1, Rmins[m+1]))
                if Rmaxs[m_max] in range(Rmins[m]+1, Rmins[m+1]):      
                    #print(Rmins[m]+1, Rmins[m+1])              
                    Hok[Rmins[m]:Rmins[m+1]] = RegionTransfer(Hsk_[Rmins[m]:Rmins[m+1]], Htk[Rmins[m]:Rmins[m+1]], k/Smax)
                    m_max += 1
            #input("")
            print('sum: ', np.sum(Htk), np.sum(Hok))            
            DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hsk), Htk, I_min, V, i, k)
            #print(Rmins)
            #print(Rmaxs)
            #DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs), Ht_, I_min, V, i, k)
            Hsk = Hok * (np.sum(Htk)/np.sum(Hok))
        Io[:,:,i] = HistMatch(Is_c, I_min, Hs, Hsk, V)
    
    return color.lab2rgb(Io)

def P(i, j, V, min_I):
    if j == math.floor((i - min_I)/V):
        return 1
    return 0

def FindPeaks(H):
    H_ = [] 
    for i in range(len(H)-1):
        H_.append(H[i+1] - H[i])
    H_mult = []
    for i in range(len(H_)-1):
        H_mult.append(H_[i]*H_[i+1])
    H2 = []
    for i in range(len(H_)-1):
        H2.append(H_[i+1] - H_[i])
    #print('FindPeaks:')
    #print(H2)
    #print(np.array(H_mult))
            
    Hr = np.logical_and(np.array(H2)>0, np.array(H_mult)<=0)

    Rmin = np.argwhere(Hr == True)
    Rmin = [r[0]+1 for r in Rmin]
    return Rmin

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
    #print('FindPeaks:')
    #print(H2)
    #print(np.array(H_mult))
            
    Hr = np.logical_and(np.array(H2)<0, np.array(H_mult)<0)

    Rmax = np.argwhere(Hr == True)
    Rmax = [r[0]+1 for r in Rmax]
    return Rmax

def RegionTransfer(Hs, Ht, wt):
    #ws = 1-wt
    ws = 1.0
    wt = 1.0
    Ho = []
    d = wt*np.std(Ht)/(ws*np.std(Hs))
    if ws*np.std(Hs)== 0:
        d = wt/ws

    for hs in Hs:
        hs_new = ((hs - ws*np.mean(Hs))*(d) + wt*np.mean(Ht))
        hs_diff = hs_new - hs
        Ho.append (hs + hs_diff*wt)
    
    #print(Hs)
    #print(Ho)
    Ho = np.array(Ho)
    return np.where(Ho < 0, 0, Ho)

def HistMatch(Is, Imin, Hs, Ho, V=1):
    Ho_ = Ho * np.sum(Hs)/np.sum(Ho)
    print('histmatch sums:', np.sum(Hs), np.sum(Ho))
    Cs = np.cumsum(Hs)
    Co = np.cumsum(Ho_)
    #create inverse fn
    Co_inv = np.repeat(len(Cs)-1, len(Cs))
    for i in range(len(Cs)):
        c = Cs[i]
        if c < Co[0]:
            Co_inv[i]=0
        else:
            for j in range(1, len(Co)):
                if c >= Co[j-1] and c <Co[j]:
                    Co_inv[i] = j
                    break
    Io = Is.copy()

    bin_idx = 0
    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
 
            bin_idx = Co_inv[int((Is[i, j] - Imin)/V)]
            #print(bin_idx)
            Io[i, j] = Imin + (bin_idx)*V
    
    #DrawHistogram(Co, Cs, Hs, Imin, V, 0, 6)
    # print(Cs[200:300])
    # print(Co[200:300])
    # print(Co_inv[200:300])
    # input("")

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

    # test = np.array([1, 1, 1, 2, 3, 5, 3, 2, 1, 1, 1, 1, 1])
    # test2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # max_idx = FindPeaks_max(test)
    # min_idx = FindPeaks(test)

    # m_max = 0
    # for m in range(len(min_idx)):
    #     if max_idx[m] in range(min_idx[m] + 1, min_idx[m+1]):
    #         #print(min_idx[m], min_idx[m+1])
    #         test2[min_idx[m]:min_idx[m+1]] = RegionTransfer(test2[min_idx[m]:min_idx[m+1]], test[min_idx[m]:min_idx[m+1]], 0.75)
    #         m_max += 1
    #         if m_max == len(max_idx):
    #             break
    # print(test2)

    # return


    file1 = 'ballerina.jpg'
    #file1 = 'bw.jpg'
    file2 = 'sunset.jpg'
    #file2 = 'ballerina.jpg'
    #file1 = 'golden_gate_5.jpg'
    #file2 = 'golden_gate_sq.jpg'
    
    I_s = io.imread(file1)
    I_t = io.imread(file2)
    I_o = ReshapeHistogram(I_s, I_t, perc=1.0, weight=True, bw=False)
    #I_o = ReshapeHistogram_fixed_bin_size(I_s, I_t, perc=1.0, bw=False)

    io.imsave('output.jpg', I_o)

    

    
    return

if __name__ == '__main__':
    main()