import numpy as np
import math
import skimage
from skimage import io, color
import matplotlib.pyplot as plt

def ReshapeHistogram(Is, It, perc):
    #convert Is, It to CIELab color space
    #Is = skimage.filters.gaussian(Is)
    #It = skimage.filters.gaussian(It)
    Is = color.rgb2lab(Is)
    It = color.rgb2lab(It)
    Io = Is.copy()

    #compute Smax
    V = 0.3
    Is_max = np.max(Is)
    It_max = np.max(It)
    Is_min = np.min(Is)
    It_min = np.min(It)
    #I = np.concatenate([Is, It])
    B = math.ceil((max(Is_max, It_max) - min(Is_min, It_min) + 1)/V)
    Bmin = 10
    Smax = math.floor(math.log2(B/Bmin))
    print(Smax)
    I_min = min(Is_min, It_min)
    
    for i in range(3):
        Is_c = Is[:,:,i]
        It_c = It[:,:,i]
        
        Hs = []
        Ht = []
        for j in range(B):
            hs = 0
            ht = 0
            for k in range(Is_c.shape[0]):
                for m in range(Is_c.shape[1]):
                    hs += P(Is_c[k, m], j, V, I_min)
            Hs.append((hs, I_min + (j)*V))

            for k in range(It_c.shape[0]):
                for m in range(It_c.shape[1]):
                    ht += P(It_c[k, m], j, V, I_min)
            Ht.append((ht, I_min + (j)*V))
        Hs_ = [h[0] for h in Hs]
        vs_ = [h[1] for h in Hs]
        Ht_ = [h[0] for h in Ht]
        vt_ = [h[1] for h in Ht]

        Hsk = np.array(Hs_)
        Htk = np.array(Ht_) 
        #print('sums: ', np.sum(Hsk), np.sum(Htk))
        Htk = Htk * (np.sum(Hsk)/np.sum(Htk)) #
        Ht_ = Htk.copy()
        
        print('sums: ', np.sum(Hsk), np.sum(Htk))
        Hok = np.array(Hs_)
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
            
            Hsk_ = Hsk.copy()
            print("weight: ", k/Smax)
            m_max = 0
            for m in range(len(Rmint)-1):
                if Rmaxt[m_max] in range(Rmint[m]+1, Rmint[m+1]):
                    Hsk_[Rmint[m]:Rmint[m+1]] = RegionTransfer(Hsk[Rmint[m]:Rmint[m+1]], Htk[Rmint[m]:Rmint[m+1]], k/Smax)
                    m_max += 1
                    if m_max == len(Rmaxt):
                        break
            #Hsk_ = Hsk_ * (np.sum(Htk)/np.sum(Hsk_))

            Rmins = FindPeaks(Hsk_)
            Rmaxs = FindPeaks_max(Hsk_)
            m_max = 0
            Hok = Hsk_.copy()
            for m in range(len(Rmins)-1):
                if Rmaxs[m_max] in range(Rmins[m]+1, Rmins[m+1]):
                    Hok[Rmins[m]:Rmins[m+1]] = RegionTransfer(Hsk_[Rmins[m]:Rmins[m+1]], Htk[Rmins[m]:Rmins[m+1]], k/Smax)
                    m_max += 1
                    if m_max == len(Rmaxs):
                        break
            
            #DrawHistogram(Hok, Hsk, Htk, I_min, V, i, k)
            print('sum: ', np.sum(Htk), np.sum(Hok))            
            #DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs_), Htk, I_min, V, i, k)
            DrawHistogram(Hok * (np.sum(Htk)/np.sum(Hok)), np.array(Hs_), Ht_, I_min, V, i, k)
            Hsk = Hok * (np.sum(Htk)/np.sum(Hok))
        Io[:,:,i] = HistMatch(Is_c, I_min, Hs_, Hsk, V)
    
    return color.lab2rgb(Io)

def P(i, j, V, min_I):
    if j == math.floor((i - min_I)/V):
        return 1
    return 0

def FindPeaks(H):
    H_ = [] 
    for i in range(len(H)-1):
        H_.append(H[i] - H[i+1])
    H_mult = []
    for i in range(len(H_)-1):
        H_mult.append(H_[i]*H_[i+1])
    H2 = []
    for i in range(len(H_)-1):
        H2.append(H_[i] - H_[i+1])
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
        H_.append(H[i] - H[i+1])

    H_mult = [] #multiplied
    for i in range(len(H_)-1):
        H_mult.append(H_[i]*H_[i+1])
    H2 = [] #second derivative
    for i in range(len(H_)-1):
        H2.append(H_[i] - H_[i+1])
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
        Ho.append (((hs - ws*np.mean(Hs))*(d) + wt*np.mean(Ht)))
    
    #print(Hs)
    #print(Ho)
    Ho = np.array(Ho)
    return np.where(Ho < 0, 0, Ho)

def HistMatch(Is, Imin, Hs, Ho, V=1):
    Ho_ = Ho * np.sum(Hs)/np.sum(Ho)
    print('histmatch sums:', np.sum(Hs), np.sum(Ho))
    Cs = np.cumsum(Hs)
    Co = np.cumsum(Ho_)

    Io = Is.copy()

    bin_idx = 0
    for i in range(Io.shape[0]):
        for j in range(Io.shape[1]):
            c = Cs[int((Is[i, j] - Imin)/V)]
            for k in range(len(Co)-1):
                if c >= Co[k] and c < Co[k+1]:
                    bin_idx = k
                    break
            Io[i, j] = Imin + (bin_idx)*V
    return Io

def DrawHistogram(H, Hs, Ht, Imin, V=1, ic=0, scale=0):
    plt.figure()
    plt.xlabel('v')

    v = []
    for i in range(len(H)):
        v.append(Imin + (i)*V)
    
    plt.plot(v, Hs)
    plt.plot(v, Ht)
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
    file2 = 'sunset.jpg'
    #file1 = 'golden_gate_5.jpg'
    #file2 = 'golden_gate_sq.jpg'
    
    I_s = io.imread(file1)
    I_t = io.imread(file2)
    I_o = ReshapeHistogram(I_s, I_t, perc=1.0)

    io.imsave('ballerina_sunset1.jpg', I_o)

    

    
    return

if __name__ == '__main__':
    main()